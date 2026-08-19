//! The deterministic builtin-head → StableHLO op map (non-distribution
//! nodes). [`Emitter::lower_node`](crate::Emitter::lower_node)'s
//! `Call{head: CallHead::Builtin(_)}` arm (and its `Const` leaf arm, for a
//! bare built-in symbol like `inf`) dispatch every non-distribution head
//! here via [`lower_builtin`].
//!
//! `lower_builtin` composes only [`Emitter`]'s already parser-validated
//! op-helper API (Task 3, plus the `slice`/`reshape`/`broadcast_in_dim`/
//! `inf` helpers this task adds alongside it) — it never builds StableHLO
//! text itself, so every emitted op inherits that layer's assembly
//! correctness.
//!
//! The map is deliberately narrow: it covers what the determiniser emits, and
//! refuses anything else rather than guessing a lowering. `broadcast(f, …)`
//! itself is handled one level up in [`Emitter::lower_broadcast`], which strips
//! the wrapper and re-enters here with `f`'s own head — every op below that is
//! elementwise therefore batches for free.
//!
//! **`*` is not `.*`.** Spec §07 "Linear algebra": "Matrix multiplication and
//! addition use the standard `*` and `+` operators." The two spellings are
//! already distinct in FlatPDL — `.*` is `broadcast(mul, …)`, plain `*` a bare
//! `mul` — so [`classify_bare_mul`]/[`lower_bare_mul`] handle a bare `mul` as one
//! of §07's four products, and the `"mul"` entry in the map below stays purely
//! elementwise. That dispatch lives in `Emitter::lower_node`, which
//! [`Emitter::lower_broadcast`] bypasses, so a `.*`-derived `mul` can never reach
//! it.
//!
//! The four products are `matrix-matrix` and `matrix-vector` (one
//! `stablehlo.dot_general`), plus the two transposed-vector orientations §07
//! defines — `transpose(a) * b` contracting to a scalar and `a * transpose(b)`
//! spreading to a matrix. Orientation is type-level only: §03 makes a transposed
//! vector a distinct TYPE, but `crate::types::mlir_type_of` maps it and a rank-1
//! array to the same `tensor<nxf32>`, so [`lower_transpose`] emits nothing for a
//! vector and the dispatch reads the orientation off the inferred types.
//!
//! **Nor is `+` `.+`.** The same split covers the other operators §07
//! "Operator-equivalent functions" gives a narrower domain than elementwise:
//! `add`/`sub` take "scalars or arrays of same shape (real or complex)",
//! `divide`/`pow` "scalars (real or complex)". The entries below are the
//! *elementwise* lowerings the dotted spellings need, so `Emitter::lower_node`
//! routes the BARE heads through [`lower_bare_arith`] first, which refuses an
//! out-of-domain operand pair rather than letting `Emitter::broadcast_pair`
//! silently answer what `.+` asks.
//!
//! A `builtin_*` primitive (`builtin_logdensityof`, `builtin_sample`,
//! `builtin_touniform`, `builtin_fromuniform`, `builtin_tonormal`,
//! `builtin_fromnormal`) or a bare distribution constructor name (`Normal`,
//! …) is §08/registry territory. As of Task 5, `Emitter::lower_node`'s `Call`
//! dispatch recognizes `builtin_logdensityof` itself and routes it to
//! `crate::registry::lower_logdensityof` *before* ever calling into this
//! module, so the catch-all "unsupported builtin head" refusal below only
//! ever sees a genuinely unknown head (or one of the other `builtin_*`
//! primitives, still unimplemented until a later task adds a matching
//! registry gate for it).

use flatppl_core::{CallHead, Node, NodeId, Scalar, Type};

use crate::emitter::{Emitter, elem_rank};
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::refuse::EmitError;

/// Lower one FlatPDL builtin call to a [`Value`]. `id` is the call (or
/// `Const`) node itself, for refusal localization — a 0-arity builtin like
/// `inf` has no argument node to blame instead. `head` is the resolved
/// builtin name; `args` its positional arguments (no op in this map reads
/// `%kwarg`/`%field` named arguments).
pub(crate) fn lower_builtin(
    e: &mut Emitter,
    id: NodeId,
    head: &str,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    match head {
        "add" => binary(e, id, args, Emitter::add),
        "sub" => binary(e, id, args, Emitter::sub),
        "mul" => binary(e, id, args, Emitter::mul),
        // §07 `divide` (real division `a / b`) — what the parser emits for `/`
        // and `./`. Distinct from §07 `div` (integer floor division, below):
        // `divide` always forces Real via `Emitter::div` (`binary_real`),
        // never the floored integer semantics.
        "divide" => binary(e, id, args, Emitter::div),
        "pow" => binary(e, id, args, Emitter::pow),
        // §07 `div`/`mod` (integer floor division ⌊a/b⌋ / floored modulo,
        // `Int` operands) — StableHLO's native `divide`/`remainder` truncate
        // toward zero, so `Emitter::floor_div`/`floor_mod` sign-correct them
        // (see their doc comments).
        "div" => binary(e, id, args, Emitter::floor_div),
        "mod" => binary(e, id, args, Emitter::floor_mod),
        "neg" => unary(e, id, args, Emitter::neg),
        "log" => unary(e, id, args, Emitter::log),
        "exp" => unary(e, id, args, Emitter::exp),
        "sqrt" => unary(e, id, args, Emitter::sqrt),
        "abs" => unary(e, id, args, Emitter::abs),
        "cos" => unary(e, id, args, Emitter::cos),
        "invlogit" => unary(e, id, args, Emitter::invlogit),
        // §07 "Elementary functions", the entries whose op or `Emitter` helper
        // already existed and only lacked a head here. Domains from §07's own
        // table; each is elementwise and shape-preserving, so it batches under
        // `broadcast` for free like every other entry in this block.
        //
        // §07 gives most of these a `complexes` domain as well, which
        // `crate::types` refuses (no complex element type) — booked once as the
        // §03 Complex row, not repeated per head.
        "sin" => unary(e, id, args, Emitter::sin),
        "floor" => unary(e, id, args, Emitter::floor),
        "ceil" => unary(e, id, args, Emitter::ceil),
        "log10" => unary(e, id, args, Emitter::log10),
        "abs2" => unary(e, id, args, Emitter::abs2),
        "asin" => unary(e, id, args, Emitter::asin),
        "acos" => unary(e, id, args, Emitter::acos),
        "acosh" => unary(e, id, args, Emitter::acosh),
        "loggamma" => unary(e, id, args, Emitter::lgamma),
        "gamma" => unary(e, id, args, Emitter::gamma),
        "atan2" => binary(e, id, args, Emitter::atan2),
        // §07's BINARY `min`/`max` ($\min(a, b)$ / $\max(a, b)$, domain
        // `reals`) — distinct from the `maximum`/`minimum` REDUCTIONS below.
        "min" => lower_binary_extremum(e, id, args, Extremum::Min),
        "max" => lower_binary_extremum(e, id, args, Extremum::Max),
        // §07 "Identities": `identity(x)` "returns `x` unchanged". Domain
        // "any", so no operand check beyond the one `lower_node` already makes
        // — a value with no tensor form refuses there, not here.
        "identity" => {
            let [a] = args_exact(id, args)?;
            e.lower_node(a)
        }
        // §06 change-of-variables heads. An open-image `pushfwd` spells THREE
        // families of head, all of which must lower or the density refuses:
        //
        // - the INVERSE (`invlogit`→`logit`, `invprobit`→`probit`,
        //   `atan`→`tan`, `expm1`→`log1p`, `sinh`→`asinh`, `asinh`→`sinh`,
        //   `tanh`→`atanh`);
        // - the head the forward's own LOG-VOLUME term spells — `cosh` for
        //   `sinh` (`log(cosh(x))`), `tanh` for `tanh` (`log(1 − tanh(x)²)`);
        // - the FORWARD itself, applied to the safe-point witness the
        //   determiniser substitutes for an out-of-image query
        //   (`probit(ifelse(gate, y, invprobit(1.0)))`) — so `invprobit`,
        //   `atan` and `expm1` are needed too, even though nothing inverts to
        //   them.
        "logit" => unary(e, id, args, Emitter::logit),
        "probit" => unary(e, id, args, Emitter::probit),
        "invprobit" => unary(e, id, args, Emitter::invprobit),
        "tan" => unary(e, id, args, Emitter::tan),
        "atan" => unary(e, id, args, Emitter::atan),
        "log1p" => unary(e, id, args, Emitter::log1p),
        "expm1" => unary(e, id, args, Emitter::expm1),
        "sinh" => unary(e, id, args, Emitter::sinh),
        "cosh" => unary(e, id, args, Emitter::cosh),
        "asinh" => unary(e, id, args, Emitter::asinh),
        "atanh" => unary(e, id, args, Emitter::atanh),
        "tanh" => unary(e, id, args, Emitter::tanh),
        // §07 `round` ("nearest integer, half to even") and §07 `real`
        // ("returns `x` for real `x`") — the pair the determiniser's
        // discrete-pushforward lattice snap emits, `real(round(x))`.
        "round" => unary(e, id, args, Emitter::round_nearest_even),
        "real" => lower_real(e, id, args),
        // §07 "Linear algebra" `transpose`/`adjoint`, domain "vectors, matrices".
        "transpose" | "adjoint" => lower_transpose(e, id, args, head),
        // §07 "Linear algebra", the entries that reduce to an `Emitter` matrix
        // helper already in the crate. Every one of the daggers §07 writes
        // ($\mathbf{A}\mathbf{A}^\dagger$, $\mathbf{x}\mathbf{x}^\dagger$,
        // $\mathbf{x}^\dagger\mathbf{A}\mathbf{x}$) is a plain transpose here,
        // for `lower_transpose`'s reason: this crate has no complex element type,
        // so over the elements it emits conjugation is the identity.
        "lower_cholesky" => lower_cholesky_head(e, id, args),
        "diag" => lower_diag(e, id, args),
        "trace" => lower_trace(e, id, args),
        "self_outer" => lower_self_outer(e, id, args),
        "row_gram" => lower_gram(e, id, args, Gram::Row),
        "col_gram" => lower_gram(e, id, args, Gram::Col),
        "quadform" => lower_quadform(e, id, args),
        // §07 "Array and table operations" `rowstack`/`colstack` (domain
        // "vector of equal-length vectors") and `addaxes` (domain "array,
        // non-negative integer, non-negative integer") — the shape
        // constructors §03 "Arrays" names as the only way to turn a
        // vector-of-vectors into a matrix.
        "rowstack" => lower_stack(e, id, args, Stack::Rows),
        "colstack" => lower_stack(e, id, args, Stack::Cols),
        "addaxes" => lower_addaxes(e, id, args),
        "ifelse" => lower_ifelse(e, id, args),
        "inf" => lower_inf(e, id, args),
        "pi" => lower_pi(e, id, args),
        "logsumexp" => lower_logsumexp(e, id, args),
        "vector" => lower_vector(e, id, args),
        "sum" => lower_sum(e, id, args),
        // §07 reductions `maximum`/`minimum` ($\max_i x_i$ / $\min_i x_i$ over
        // a real array) — NOT §07's binary `max`/`min`, which this map does not
        // lower.
        "maximum" => lower_extremum(e, id, args, Extremum::Max),
        "minimum" => lower_extremum(e, id, args, Extremum::Min),
        "fill" => lower_fill(e, id, args),
        "get0" => lower_get(e, id, args, 0),
        "get" => lower_get(e, id, args, 1),
        // §04 "Multi-axis aggregation" — the einsum-style contraction, and what
        // the surface `:=` sugar desugars to. `crate::aggregate` owns the whole
        // lowering (it binds the body's axis-indexed operands before walking the
        // body, so it cannot be composed out of the elementwise entries above).
        "aggregate" => crate::aggregate::lower_aggregate(e, id, args),
        // §04 "Metric-aware Einstein summation" — refused with its own reason
        // rather than falling through to the generic unknown-head message; see
        // `aggregate::metricsum_refusal`.
        "metricsum" => Err(crate::aggregate::metricsum_refusal(id)),
        "in" => lower_in(e, id, args),
        // §07 comparison functions `lt`/`gt`/`le`/`ge` ($a < b$, $a > b$, $a \le b$,
        // $a \ge b$ over `reals`). The inclusive pair is the image gate's vocabulary
        // for a CLOSED finite endpoint (`determinizer::invert`): `pushfwd(exp, Gamma)`
        // has image [1, ∞), which no §03 set spells — `interval(1, inf)` would, but
        // its lowering is the product `(v − lo)·(hi − v) ≥ 0`, and at `v = lo` with an
        // infinite `hi` that is `0 · inf` = NaN, i.e. FALSE at the one endpoint the
        // closed bound exists to admit.
        "lt" => lower_compare(e, id, args, "LT"),
        "gt" => lower_compare(e, id, args, "GT"),
        "le" => lower_compare(e, id, args, "LE"),
        "ge" => lower_compare(e, id, args, "GE"),
        // §07 "Comparison functions" `equal`/`unequal` ($a = b$, $a \neq b$).
        // Their domain is DISCRETE ONLY — see [`lower_exact_compare`].
        "equal" => lower_exact_compare(e, id, args, "EQ"),
        "unequal" => lower_exact_compare(e, id, args, "NE"),
        "land" => lower_land(e, id, args, Connective::And),
        // §07 "Logical operators" `lor`/`lxor`/`lnot` (`a || b`, exclusive or,
        // `!a`, all over `booleans`) — `stablehlo.or`/`xor`/`not` over the same
        // `i1` predicates `land` takes, with the same operand rule.
        "lor" => lower_land(e, id, args, Connective::Or),
        "lxor" => lower_land(e, id, args, Connective::Xor),
        "lnot" => lower_lnot(e, id, args),
        "iszero" => lower_iszero(e, id, args),
        // §07 "Scalar predicates" `isfinite`/`isinf`/`isnan`.
        "isfinite" => lower_finiteness(e, id, args, Finiteness::Finite),
        "isinf" => lower_finiteness(e, id, args, Finiteness::Inf),
        "isnan" => lower_finiteness(e, id, args, Finiteness::Nan),
        // `record(...)` is not a tensor — handled structurally by the mode
        // builder (a record-typed model input's fields become separate
        // tensor args), never reached here in a well-formed lowering.
        "record" => Err(EmitError::at(id, "record has no tensor form")),
        // A `load_data` listed in `inputs` is pre-bound to its argument by the
        // mode builder and never lowered. Reaching here means it is used as one
        // monolithic value while its valueset is an aggregate, which the
        // per-column destructuring cannot supply.
        "load_data" => Err(EmitError::at(
            id,
            "a load_data input whose valueset is a table or record has no single \
             tensor form; read it column-wise (`data.y`) — one argument per column",
        )),
        other => Err(EmitError::at(
            id,
            format!("unsupported builtin head '{other}'"),
        )),
    }
}

// ---- arity-checked leaf combinators ----------------------------------------

/// Destructure `args` into exactly `N` positional arguments, or refuse.
fn args_exact<const N: usize>(id: NodeId, args: &[NodeId]) -> Result<[NodeId; N], EmitError> {
    <[NodeId; N]>::try_from(args)
        .map_err(|_| EmitError::at(id, format!("expected {N} argument(s), got {}", args.len())))
}

fn unary<'m>(
    e: &mut Emitter<'m>,
    id: NodeId,
    args: &[NodeId],
    op: fn(&mut Emitter<'m>, &Value) -> Value,
) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    let a = e.lower_node(a)?;
    Ok(op(e, &a))
}

/// Refuse an operand pair whose ORIENTATION differs — a column (a rank-1
/// `Array`) against a row (a `TVector`).
///
/// The elementwise, compare and select paths all reconcile through `MlirTy`,
/// where a `TVector{n}` and a rank-1 `Array[n]` are the same `tensor<nxf32>` — so
/// §03's distinction ("In addition, transposed vectors are a distinct type in
/// FlatPPL") is invisible to `require_broadcastable` / `Emitter::compare` /
/// `Emitter::select`, and a row/column mix lowered silently. This reads the
/// INFERRED types instead, the way [`classify_bare_mul`] does, and it is why the
/// check lives in the `crate::ops` callers rather than in
/// [`require_broadcastable`], which only ever sees `MlirTy`.
///
/// Deliberately NOT the whole of [`ArithShape::differs_from`], which also reports
/// a scalar against an array: the dotted spellings exist to broadcast exactly
/// that (`s .+ v`), and the determiniser's synthesized `mul(literal, vector)`
/// idiom reaches [`binary`] too, so refusing that pair would break far more than
/// it fixes. Extent mismatches within one orientation stay
/// [`require_broadcastable`]'s job.
fn require_same_orientation(
    e: &Emitter,
    id: NodeId,
    a: NodeId,
    b: NodeId,
) -> Result<(), EmitError> {
    let (sa, sb) = (arith_shape(e, a), arith_shape(e, b));
    if matches!(
        (&sa, &sb),
        (ArithShape::Array(_), ArithShape::TVector(_))
            | (ArithShape::TVector(_), ArithShape::Array(_))
    ) {
        let shape = |n: NodeId| match e.type_of(n) {
            Some(t) => format!("{t:?}"),
            None => "unknown".to_string(),
        };
        return Err(EmitError::at(
            id,
            format!(
                "operands have different orientation: {} against {} — §03 makes a transposed \
                 vector a distinct type from a one-dimensional array, so combining a row with \
                 a column elementwise is not defined even though both are `tensor<nxf32>`. \
                 Transpose one of them to match",
                shape(a),
                shape(b)
            ),
        ));
    }
    Ok(())
}

fn binary<'m>(
    e: &mut Emitter<'m>,
    id: NodeId,
    args: &[NodeId],
    op: fn(&mut Emitter<'m>, &Value, &Value) -> Value,
) -> Result<Value, EmitError> {
    let [a, b] = args_exact(id, args)?;
    // Orientation first, from the inferred types, while the node ids are still in
    // hand — `require_broadcastable` below sees only `MlirTy`, which cannot tell a
    // row from a column.
    require_same_orientation(e, id, a, b)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    require_broadcastable(id, &a, &b)?;
    Ok(op(e, &a, &b))
}

/// Refuse an operand pair `Emitter::broadcast_pair` would panic on. The
/// `Emitter::binary`/`compare`/`select` helpers are infallible and have no
/// `Result` to carry an [`EmitError`], so the check belongs in their callers here
/// — and there are THREE, every one of which must call it or a pair with no
/// broadcast form aborts the process instead of refusing:
///
/// - [`binary`] (every arity-2 arithmetic head),
/// - [`lower_compare`] (`Emitter::compare` reconciles its operands the same way),
/// - [`lower_ifelse`] (`Emitter::select` broadcasts the branch pair, AND both
///   branches against the predicate's shape — so it checks two pairings).
///
/// An earlier round guarded only [`binary`] while claiming the panic was closed;
/// `compare` and `select` kept aborting on ordinary surface FlatPPL
/// (`ifelse(p < q, m, v)` with a matrix and a vector branch).
///
/// Enumerates the pairs `broadcast_pair` HANDLES and refuses everything else,
/// rather than listing the pairs it panics on: `MlirTy` has four variants, so a
/// `Key` or `Tuple` operand reaches its `(ta, tb) => panic!` arm exactly as a
/// rank mismatch does. Matching positively also means a variant added later
/// refuses by default instead of silently acquiring a panic path.
fn require_broadcastable(id: NodeId, a: &Value, b: &Value) -> Result<(), EmitError> {
    let refuse = |why: &str| {
        Err(EmitError::at(
            id,
            format!(
                "elementwise operands do not broadcast: {:?} against {:?} — {why}",
                a.ty, b.ty
            ),
        ))
    };
    // An rng-state key and a tuple have no arithmetic form at all, equal types or
    // not: `broadcast_pair`'s equal-type fast path would hand such a pair straight
    // through to an arithmetic op that cannot mean anything on it.
    let unarithmetic = |t: &MlirTy| matches!(t, MlirTy::Key | MlirTy::Tuple(_));
    if unarithmetic(&a.ty) || unarithmetic(&b.ty) {
        return refuse(
            "an rng-state key and a tuple have no elementwise arithmetic form (spec §07's \
             rng state is threaded, never computed on)",
        );
    }
    match (&a.ty, &b.ty) {
        // A scalar operand broadcasts against any rank.
        (MlirTy::Scalar, _) | (_, MlirTy::Scalar) => Ok(()),
        (MlirTy::Ranked(da), MlirTy::Ranked(db)) => {
            let compatible = da.len() == db.len()
                && da.iter().zip(db.iter()).all(|(&x, &y)| {
                    matches!(
                        (x, y),
                        (Some(m), Some(n)) if m == n)
                        || matches!(
                            (x, y),
                            (Some(1), Some(_)) | (Some(_), Some(1)) | (None, None)
                        )
                });
            if compatible {
                Ok(())
            } else {
                refuse(
                    "§04 broadcasting needs equal rank with each axis pair equal or size 1 \
                     (a matrix product is the non-elementwise `*`, not `.*`)",
                )
            }
        }
        _ => refuse("no broadcast form for this shape pair"),
    }
}

/// What a BARE `mul` (surface `*`, spec §07 "Linear algebra") means for the
/// shapes of its operands.
enum BareMul {
    /// A rank-2 lhs against a rank-2 or rank-1 rhs: one `stablehlo.dot_general`
    /// (`matrix-matrix` / `matrix-vector` in §07's `mul` row).
    MatrixProduct,
    /// `transposed-vector–vector` — §07: "the product of a transposed vector and
    /// a non-transposed vector is a scalar".
    InnerProduct,
    /// `vector–transposed-vector` — §07: "The product of a non-transposed vector
    /// and a transposed vector is a matrix".
    OuterProduct,
    /// A row vector against a matrix, `row[k] · [k, n] → row[n]`. Not in §07's
    /// `mul` row as of design `9e35262`; flatppl-design#77 (pending owner review)
    /// adds `transposed-vector–matrix` to it.
    RowMatrixProduct,
    /// At least one operand is a scalar (or has no inferred type): the ordinary
    /// elementwise multiply, which broadcasts the scalar side.
    Elementwise,
    /// Both operands are non-scalar but the pair is not a product §07 admits — so
    /// lowering it elementwise would answer a different question than the model
    /// asked.
    Undefined,
}

/// Classify a bare `mul`'s operands by their inferred FlatPDL types — the same
/// information `infer`'s `mul_type` reads — so no operand is lowered to decide.
///
/// The admitted set is §07's `mul` row verbatim: "scalars, matrix-matrix,
/// matrix-vector, scalar-matrix, scalar-vector, transposed-vector–vector,
/// vector–transposed-vector". Orientation is load-bearing, so this reads
/// [`arith_shape`] (which keeps `Array` and `TVector` distinct, per §03's "In
/// addition, transposed vectors are a distinct type in FlatPPL") rather than a
/// bare rank.
///
/// Everything else is [`BareMul::Undefined`], and `infer` independently agrees —
/// `mul_type` returns `Type::Deferred` for each (measured):
/// - vector·vector and TVector·TVector: §07 gives `*` a vector meaning ONLY
///   through a transpose, and only in the two mixed orientations;
/// - matrix·TVector and TVector·matrix: the row lists `matrix-vector` but no
///   `vector-matrix`, and it distinguishes transposed from non-transposed exactly
///   where orientation decides the result. §03's "the term vector will represent
///   both … unless noted otherwise" could be read as widening `matrix-vector` to
///   cover a transposed rhs, but `[m,k] · row[k]` does not conform dimensionally,
///   so that reading contradicts the maths and is not taken. A row-vector–matrix
///   product IS sound maths, and admitting it would be a spec-row change rather
///   than a guard relaxation;
/// - rank 3 and above: not enumerated at all.
///
/// An operand whose type is absent (a freshly synthesized determiniser node
/// before re-inference) or scalar classifies as [`BareMul::Elementwise`], which
/// is what keeps the determiniser's own `mul(literal, vector)` idiom lowering.
fn classify_bare_mul(e: &Emitter, args: &[NodeId]) -> BareMul {
    use ArithShape::*;
    let [a, b] = match <[NodeId; 2]>::try_from(args) {
        Ok(pair) => pair,
        // Wrong arity: let `binary`'s `args_exact` produce the arity message.
        Err(_) => return BareMul::Elementwise,
    };
    match (arith_shape(e, a), arith_shape(e, b)) {
        (Scalar, _) | (_, Scalar) | (Unknown, _) | (_, Unknown) => BareMul::Elementwise,
        (Array(da), Array(db)) if da.len() == 2 && (db.len() == 2 || db.len() == 1) => {
            BareMul::MatrixProduct
        }
        (TVector(da), Array(db)) if da.len() == 1 && db.len() == 1 => BareMul::InnerProduct,
        (Array(da), TVector(db)) if da.len() == 1 && db.len() == 1 => BareMul::OuterProduct,
        // Row-vector against a matrix. AHEAD OF THE SPEC ROW: see
        // [`lower_vector_product`]'s doc for why this is admitted while the
        // mirrored `matrix * row` below stays refused.
        (TVector(da), Array(db)) if da.len() == 1 && db.len() == 2 => BareMul::RowMatrixProduct,
        _ => BareMul::Undefined,
    }
}

/// Lower a bare `mul` — the surface `*`. Routes by [`classify_bare_mul`]:
/// a matrix product to [`lower_matrix_product`], an undefined non-scalar pair to
/// a refusal, and everything else to the elementwise `mul` in [`lower_builtin`].
///
/// Reached ONLY from `Emitter::lower_node`'s dispatch, never from
/// `Emitter::lower_broadcast` — so the elementwise `.*` spelling
/// (`broadcast(mul, …)`) is not classified here and keeps its own meaning.
///
/// All four §07 products lower here: `matrix-matrix` and `matrix-vector` through
/// [`lower_matrix_product`], and the two transposed-vector orientations through
/// [`Emitter::inner_product`] / [`Emitter::outer_product`].
pub(crate) fn lower_bare_mul(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    match classify_bare_mul(e, args) {
        BareMul::MatrixProduct => lower_matrix_product(e, id, args),
        BareMul::InnerProduct => lower_vector_product(e, id, args, VectorProduct::Inner),
        BareMul::OuterProduct => lower_vector_product(e, id, args, VectorProduct::Outer),
        BareMul::RowMatrixProduct => lower_vector_product(e, id, args, VectorProduct::RowMatrix),
        BareMul::Elementwise => binary(e, id, args, Emitter::mul),
        BareMul::Undefined => {
            let shape = |n: NodeId| match e.type_of(n) {
                Some(t) => format!("{t:?}"),
                None => "unknown".to_string(),
            };
            Err(EmitError::at(
                id,
                format!(
                    "`*` has no meaning for these operand shapes: {} against {} — §07 \
                     \"Linear algebra\" gives `mul` the domain \"scalars, matrix-matrix, \
                     matrix-vector, scalar-matrix, scalar-vector, transposed-vector–vector, \
                     vector–transposed-vector\", which admits a vector pair only in a mixed \
                     transposed orientation and no rank-3 operand at all. Write `.*` for an \
                     elementwise product, `transpose(a) * b` for an inner product, or \
                     `a * transpose(b)` for an outer product",
                    shape(args[0]),
                    shape(args[1])
                ),
            ))
        }
    }
}

/// Lower §07 "Linear algebra"'s `transpose` / `adjoint`, whose domain is
/// "vectors, matrices".
///
/// A MATRIX swaps its two axes: `stablehlo.transpose … dims = [1, 0]`.
///
/// A VECTOR is a no-op at the tensor level, and that is the substantive point.
/// §07 says "The transpose of a vector is a transposed vector …, not a
/// single-row matrix", and §03 makes that transposed vector a distinct TYPE — but
/// it has no distinct tensor form, since `crate::types::mlir_type_of` maps both a
/// rank-1 `Array` and a `TVector` to `tensor<nxf32>`. So the transposition is
/// carried entirely by the inferred type, which is where `crate::ops`'s `mul`
/// dispatch reads orientation from; emitting a `stablehlo.transpose … dims = [0]`
/// would be a legal identity and pure noise. The operand's `Value` passes
/// straight through.
///
/// This also settles §07's "`transpose` and `adjoint` are self-inverse" for
/// vectors without a fold: `transpose(transpose(v))` emits nothing either time.
/// For a matrix the two `stablehlo.transpose` ops ARE both emitted — an
/// involution left to the consuming compiler to fold rather than pattern-matched
/// here, since a peephole over `%N = transpose(transpose(x))` would be this
/// crate's first such rewrite and buys nothing XLA does not already do.
///
/// `adjoint` is the CONJUGATE transpose, but this crate has no complex element
/// type at all — [`crate::mlir::ElemKind`] is Real/Int/Bool only, and
/// `mlir_type_of` refuses a `Complex` scalar outright — so over the elements this
/// crate emits the conjugation is the identity, and `adjoint` is exactly
/// `transpose`. A complex operand cannot reach here today; if a complex element
/// type is ever added, this arm must grow a conjugation before the permutation.
///
/// Rank 3 and above refuses: §07's domain is vectors and matrices, and there is
/// no canonical permutation for a higher-rank array.
fn lower_transpose(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    head: &str,
) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    let v = e.lower_node(a)?;
    match &v.ty {
        // A vector or a transposed vector: the distinction is type-level only.
        MlirTy::Ranked(d) if d.len() == 1 => Ok(v),
        MlirTy::Ranked(d) if d.len() == 2 => Ok(e.transpose(&v, &[1, 0])),
        other => Err(EmitError::at(
            id,
            format!(
                "`{head}` has no lowering for {other:?} — §07 \"Linear algebra\" gives it the \
                 domain \"vectors, matrices\", so a rank-3 or higher operand has no transpose"
            ),
        )),
    }
}

// ---- §07 linear algebra --------------------------------------------------------

/// A statically-shaped rank-2 operand's dimensions, or a refusal. Every §07
/// linear-algebra lowering below needs the two extents as NUMBERS — to check
/// squareness, to size a result, or to reject a rank the helper would panic on
/// — so a dynamic (`?`) axis refuses here rather than emitting a module whose
/// shape contract cannot be checked.
fn matrix_dims(id: NodeId, v: &Value, head: &str) -> Result<(u64, u64), EmitError> {
    match &v.ty {
        MlirTy::Ranked(d) if d.len() == 2 => match (d[0], d[1]) {
            (Some(m), Some(n)) => Ok((m, n)),
            _ => Err(EmitError::at(
                id,
                format!(
                    "`{head}`: a dynamic matrix axis has no lowering, got {:?}",
                    v.ty
                ),
            )),
        },
        other => Err(EmitError::at(
            id,
            format!(
                "`{head}`: §07 \"Linear algebra\" gives this a matrix domain, so a rank-2 \
                 operand is required, got {other:?}"
            ),
        )),
    }
}

/// [`matrix_dims`] plus §07's SQUARE requirement, returning the single extent.
fn square_dim(id: NodeId, v: &Value, head: &str) -> Result<u64, EmitError> {
    let (m, n) = matrix_dims(id, v, head)?;
    if m != n {
        return Err(EmitError::at(
            id,
            format!(
                "`{head}`: §07 \"Linear algebra\" gives this the domain \"square matrices\", \
                 got {m}x{n}"
            ),
        ));
    }
    Ok(m)
}

/// Refuse a non-`Real` operand for the two helpers that hardcode a real element
/// type. [`Emitter::cholesky`] renders `stablehlo.cholesky` at the operand's own
/// kind (an integer operand emits an op IREE rejects) and [`Emitter::diag`]
/// renders its iota/mask matrices and its reduction identity as floats
/// unconditionally — so an `Int` or `Bool` operand would mis-emit rather than
/// lower. §07's domains here (positive definite / matrices used through a
/// Cholesky or a diagonal) are real anyway; §03's `booleans ⊂ integers ⊂ reals`
/// means the caller can convert first.
fn require_real_matrix(id: NodeId, v: &Value, head: &str) -> Result<(), EmitError> {
    if v.elem == ElemKind::Real {
        return Ok(());
    }
    Err(EmitError::at(
        id,
        format!(
            "`{head}`: only a real matrix is supported, got {:?} — the underlying lowering \
             emits float-typed index and identity constants. Convert to reals first",
            v.elem
        ),
    ))
}

/// §07 `lower_cholesky(A)` — "lower-triangular $\mathbf{L}$ with
/// $\mathbf{A} = \mathbf{L}\mathbf{L}^\dagger$ and positive diagonal entries",
/// domain "positive definite `A`". One `stablehlo.cholesky` with `lower = true`.
///
/// Positive-definiteness is §07's precondition on the CALLER, not something a
/// static emitter can check: `stablehlo.cholesky` is documented to produce
/// implementation-defined values for a non-positive-definite operand, exactly as
/// §07 leaves it a domain condition.
fn lower_cholesky_head(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a_id] = args_exact(id, args)?;
    let a = e.lower_node(a_id)?;
    square_dim(id, &a, "lower_cholesky")?;
    require_real_matrix(id, &a, "lower_cholesky")?;
    Ok(e.cholesky(&a))
}

/// §07 `diag(A, k)` — "extracts the $k$th diagonal of $\mathbf{A}$ as a vector
/// … when called as `diag(A)`, `k` defaults to `0`".
///
/// PARTIAL against that entry, in two ways, each refused rather than
/// approximated:
///
/// - **`k` must be the literal `0`.** [`Emitter::diag`] masks with
///   `row == col`; a super- or sub-diagonal needs a shifted mask and a shorter
///   result, which is a different lowering rather than a parameter of this one.
/// - **`A` must be SQUARE**, though §07's domain is "matrices". `Emitter::diag`
///   row-sums an `n`-column mask, so on an `m`x`n` operand with `m > n` it
///   returns `m` entries — zeros for the rows the diagonal never reaches —
///   instead of the `min(m, n)` §07 defines. Refusing keeps that from being
///   silently answered.
fn lower_diag(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let (a_id, k_id) = match args {
        [a] => (*a, None),
        [a, k] => (*a, Some(*k)),
        _ => {
            return Err(EmitError::at(
                id,
                format!("`diag`: expected 1 or 2 argument(s), got {}", args.len()),
            ));
        }
    };
    if let Some(k) = k_id {
        let k_val = literal_index(e, id, k).map_err(|_| {
            EmitError::at(
                id,
                "`diag`: the diagonal offset `k` must be an integer literal",
            )
        })?;
        if k_val != 0 {
            return Err(EmitError::at(
                id,
                format!(
                    "`diag`: only the MAIN diagonal (`k = 0`, §07's default) lowers, got k = \
                     {k_val} — a §07 super- or sub-diagonal needs a shifted mask and a shorter \
                     result than this lowering produces"
                ),
            ));
        }
    }
    let a = e.lower_node(a_id)?;
    square_dim(id, &a, "diag")?;
    require_real_matrix(id, &a, "diag")?;
    Ok(e.diag(&a))
}

/// §07 `trace(A)` — $\mathrm{tr}(\mathbf{A})$ over "square matrices", the sum
/// of [`Emitter::diag`]'s extraction. Same square/real limits as
/// [`lower_diag`], and for the same reasons.
fn lower_trace(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a_id] = args_exact(id, args)?;
    let a = e.lower_node(a_id)?;
    square_dim(id, &a, "trace")?;
    require_real_matrix(id, &a, "trace")?;
    let d = e.diag(&a);
    Ok(e.reduce_sum(&d))
}

/// §07 `self_outer(x)` — "$\mathbf{x} \cdot \mathbf{x}^\dagger$ (outer
/// product)", domain "vectors". One [`Emitter::outer_product`] of the operand
/// against itself, giving `[n, n]`.
///
/// Orientation is not read: §07 gives the entry ONE domain ("vectors") and one
/// result, and §03 makes a transposed vector the same tensor, so both spellings
/// lower to the same square matrix.
fn lower_self_outer(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [x_id] = args_exact(id, args)?;
    let x = e.lower_node(x_id)?;
    match &x.ty {
        MlirTy::Ranked(d) if d.len() == 1 => {}
        other => {
            return Err(EmitError::at(
                id,
                format!(
                    "`self_outer`: §07 \"Linear algebra\" gives it the domain \"vectors\", so a \
                     rank-1 operand is required, got {other:?}"
                ),
            ));
        }
    }
    Ok(e.outer_product(&x, &x))
}

/// Which §07 Gram matrix a head builds.
#[derive(Clone, Copy)]
enum Gram {
    /// `row_gram(A)` — $\mathbf{A} \mathbf{A}^\dagger$, `[m, n] -> [m, m]`.
    Row,
    /// `col_gram(A)` — $\mathbf{A}^\dagger \mathbf{A}$, `[m, n] -> [n, n]`.
    Col,
}

/// §07 `row_gram(A)` / `col_gram(A)`, domain "matrices" — one
/// [`Emitter::transpose`] and one [`Emitter::matmat`]. Kind-polymorphic: the
/// product runs through `Emitter::dot_contract`, which widens to the operands'
/// common kind, so an integer matrix keeps an integer Gram.
fn lower_gram(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Gram,
) -> Result<Value, EmitError> {
    let head = match which {
        Gram::Row => "row_gram",
        Gram::Col => "col_gram",
    };
    let [a_id] = args_exact(id, args)?;
    let a = e.lower_node(a_id)?;
    matrix_dims(id, &a, head)?;
    let at = e.transpose(&a, &[1, 0]);
    Ok(match which {
        Gram::Row => e.matmat(&a, &at),
        Gram::Col => e.matmat(&at, &a),
    })
}

/// §07 `quadform(A, x)` — "$\mathbf{x}^\dagger \mathbf{A} \mathbf{x}$", domain
/// "square `A`, vector `x`". Associated as $\mathbf{x}^\dagger(\mathbf{A}
/// \mathbf{x})$: one [`Emitter::matvec`] then one [`Emitter::inner_product`],
/// which is two `dot_general`s rather than the three a left-to-right
/// association would need, and gives §07's scalar result.
fn lower_quadform(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a_id, x_id] = args_exact(id, args)?;
    let a = e.lower_node(a_id)?;
    let x = e.lower_node(x_id)?;
    let n = square_dim(id, &a, "quadform")?;
    let len = match &x.ty {
        MlirTy::Ranked(d) if d.len() == 1 => d[0],
        other => {
            return Err(EmitError::at(
                id,
                format!(
                    "`quadform`: §07 gives `x` the domain \"vector\", so a rank-1 operand is \
                     required, got {other:?}"
                ),
            ));
        }
    };
    if len != Some(n) {
        return Err(EmitError::at(
            id,
            format!("`quadform`: `x` must have length {n} to match the {n}x{n} `A`, got {len:?}"),
        ));
    }
    let ax = e.matvec(&a, &x);
    Ok(e.inner_product(&x, &ax))
}

/// Which matrix a §07 stack constructor builds out of its argument's vectors.
#[derive(Clone, Copy)]
enum Stack {
    /// `rowstack(vs)` — "a matrix whose rows are the vectors in `vs`".
    Rows,
    /// `colstack(vs)` — "a matrix whose columns are the vectors in `vs`".
    Cols,
}

/// Lower §07 "Array and table operations"'s `rowstack(vs)` / `colstack(vs)`,
/// whose argument is "a vector of vectors, all of the same length".
///
/// **`rowstack` emits nothing**, and that is the substantive point. §03
/// "Arrays" keeps a vector-of-vectors distinct from a matrix — "Vectors of
/// vectors are not interpreted as matrices implicitly, but can be turned into
/// matrices explicitly using `rowstack` or `colstack`" — but the distinction is
/// type-level only here, exactly as a transposed vector's is in
/// [`lower_transpose`]: `crate::types::mlir_type_of` FLATTENS a nested `Array`
/// element chain into one tensor shape, so `vs` has already lowered to the
/// `[n, m]` tensor whose row `i` is `vs[i]`. The outer axis leads in both
/// producers — `Emitter::vector` concatenates along dim 0, and
/// `types::flatten_elem` appends the inner dims after the outer for an ABI
/// input — so §07's row order is already the one emitted and the operand's
/// `Value` passes straight through. Emitting a rank-2 identity
/// `stablehlo.transpose … dims = [0, 1]` would be pure noise.
///
/// **`colstack` is that matrix transposed**, one `stablehlo.transpose … dims =
/// [1, 0]`. §07's own worked example is the check: `colstack([[1, 2, 3], [4, 5,
/// 6]])` is the 3x2 matrix `[[1, 4], [2, 5], [3, 6]]`, which is `rowstack`'s
/// 2x3 `[[1, 2, 3], [4, 5, 6]]` with its axes swapped.
///
/// A non-rank-2 operand refuses: rank 1 is a vector of SCALARS and rank 3 a
/// vector of matrices, neither of which is §07's "vector of vectors". A RAGGED
/// container refuses one level down, in [`lower_vector`] — inference types
/// `[[1.0, 2.0], [3.0]]` as an array of `%any` rather than reporting it, so the
/// emitter is the first layer to see it, and `lower_vector`'s
/// identical-`MlirTy` check is where it is caught.
///
/// A MATRIX argument refuses too, and that check cannot be the one above: §03's
/// "Vectors of vectors are not interpreted as matrices implicitly" has a converse
/// — a matrix standing in for a vector of vectors is equally unsanctioned — but
/// the two have the SAME lowered tensor, so only the inferred type can tell them
/// apart ([`require_vector_container`]).
///
/// ORIENTATION: a container whose elements are uniformly TRANSPOSED vectors is
/// accepted. This is a CHOICE, not a settled reading. §03 "Arrays" states the
/// blanket it rests on — "The term vector will represent both non-transposed
/// vectors (one-dimensional arrays) and transposed vectors in the following,
/// unless noted otherwise" — and §07's entry names where the argument's vectors
/// go (rows / columns), which fixes the result whatever their own orientation, so
/// no number depends on the choice. But §03's array definition ("collections of
/// scalar values (real, integer, boolean and complex values) or arrays") does not
/// list transposed vectors, so an array OF transposed vectors is a value type the
/// spec leaves unspecified rather than grants; `infer` types the enclosing
/// `rowstack` `%deferred` rather than accepting it. Refusing, as `addaxes` does,
/// would have been equally defensible; accepting keeps the blanket's plain
/// reading, and `rowstack([transpose(v1), transpose(v2)])` emits the
/// byte-identical module to the plain spelling. A container that MIXES the two
/// refuses ([`require_uniform_orientation`]).
fn lower_stack(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    kind: Stack,
) -> Result<Value, EmitError> {
    let head = match kind {
        Stack::Rows => "rowstack",
        Stack::Cols => "colstack",
    };
    let [vs_id] = args_exact(id, args)?;
    require_vector_container(e, id, vs_id, head)?;
    require_uniform_orientation(e, id, vs_id, head)?;
    let vs = e.lower_node(vs_id)?;
    if !matches!(&vs.ty, MlirTy::Ranked(dims) if dims.len() == 2) {
        return Err(EmitError::at(
            id,
            format!(
                "`{head}` has no lowering for {:?} — §07 \"Array and table operations\" gives it \
                 the domain \"vector of equal-length vectors\", so the argument must lower to a \
                 rank-2 tensor (a rank-1 operand is a vector of scalars, a rank-3 one a vector \
                 of matrices)",
                vs.ty
            ),
        ));
    }
    Ok(match kind {
        Stack::Rows => vs,
        Stack::Cols => e.transpose(&vs, &[1, 0]),
    })
}

/// Refuse a `rowstack`/`colstack` argument that is a MATRIX (or any rank-2+
/// array) rather than §07's "vector of equal-length vectors".
///
/// The rank-2 check in [`lower_stack`] is on the LOWERED tensor, which cannot see
/// the difference: `types::mlir_type_of` flattens a nested element chain, so a
/// `[2]`-of-`[2]` container and a `[2, 2]` matrix are both `tensor<2x2xf32>`. That
/// is exactly what makes `rowstack` a no-op for a real container — and exactly
/// what would make `rowstack(matrix)` a silent identity and `colstack(matrix)` a
/// silent transpose of an operand §07 gives the constructor no meaning for. §03
/// "Arrays" states the direction the spec does not sanction — "Vectors of vectors
/// are not interpreted as matrices implicitly, but can be turned into matrices
/// explicitly using `rowstack` or `colstack`" — and the converse is no more
/// granted than the stated one.
///
/// Decided on the INFERRED type, the only layer that still distinguishes them,
/// and `infer` independently agrees: `rowstack_type` (`crates/infer/src/ops.rs`)
/// matches only a rank-1 array whose element is itself an array, so `rowstack(A)`
/// on a matrix `A` is already `%deferred`.
///
/// Refuses ONLY a provably rank-2-or-higher `Array`. Everything else falls
/// through to the checks that own it:
/// - an outer `TVector` — a transposed container, rank-1, accepted (see
///   [`lower_stack`]'s orientation note);
/// - a rank-1 `Array` of scalars — [`lower_stack`]'s lowered-rank refusal;
/// - a rank-1 `Array` of `%any` (what inference gives a RAGGED or
///   MIXED-orientation container) — [`lower_vector`]'s ragged refusal or
///   [`require_uniform_orientation`];
/// - no inferred type, `%deferred`, `%any` (a freshly synthesized determiniser
///   node) — the lowered-rank check, as before.
fn require_vector_container(
    e: &Emitter,
    id: NodeId,
    vs_id: NodeId,
    head: &str,
) -> Result<(), EmitError> {
    // A use site reaches its binding through one `(%ref self x)` hop, and only
    // the binding's rhs carries the inferred type in that case.
    let ty = e
        .type_of(vs_id)
        .or_else(|| e.type_of(e.resolve_ref_one(vs_id)));
    let Some(Type::Array { shape, .. }) = ty else {
        return Ok(());
    };
    if shape.len() < 2 {
        return Ok(());
    }
    Err(EmitError::at(
        id,
        format!(
            "`{head}`'s argument is a rank-{} array, not a vector of vectors — §07 \"Array and \
             table operations\" gives `{head}` the domain \"vector of equal-length vectors\", and \
             §03 \"Arrays\" says vectors of vectors \"are not interpreted as matrices implicitly, \
             but can be turned into matrices explicitly using `rowstack` or `colstack`\", so a \
             matrix standing in for the container is not sanctioned either. The two have the same \
             tensor form, so this would otherwise lower silently — an identity for `rowstack`, a \
             transpose for `colstack`. A matrix is already stacked; pass the vectors themselves",
            shape.len()
        ),
    ))
}

/// Refuse a `rowstack`/`colstack` container that MIXES orientations — one
/// element a rank-1 `Array` (a column), another a `TVector` (a row).
///
/// §03 "Arrays" makes an array's elements one type ("fixed-size, ordered,
/// n-dimensional collections of scalar values … or arrays"), and a transposed
/// vector is a distinct type from a one-dimensional array, so a mixed container
/// is not a well-typed FlatPPL array at all. Inference agrees without saying so:
/// it types `[v1, transpose(v2)]` as an array of `%any` and the enclosing
/// `rowstack` `%deferred`, so nothing upstream reports it — and both elements
/// lower to the same `tensor<nxf32>`, so [`lower_vector`]'s identical-`MlirTy`
/// check accepts the pair and the stack would lower silently. Same
/// accepts-invalid trap as [`require_same_orientation`]'s, and closed the same
/// way: on the INFERRED types, which is the only layer that can still tell a row
/// from a column.
///
/// Only a literal `vector(...)` container is inspected — that is the one shape
/// whose per-element types are visible here. A container reached any other way
/// (an ABI input, a `partition` result) has a single element type by
/// construction and cannot mix.
fn require_uniform_orientation(
    e: &Emitter,
    id: NodeId,
    vs_id: NodeId,
    head: &str,
) -> Result<(), EmitError> {
    let vs_id = e.resolve_ref_one(vs_id);
    let Node::Call(c) = e.node(vs_id) else {
        return Ok(());
    };
    if !matches!(c.head, CallHead::Builtin(sym) if e.resolve(sym) == "vector") {
        return Ok(());
    }
    let (mut saw_column, mut saw_row) = (false, false);
    for &el in c.args.iter() {
        match arith_shape(e, el) {
            ArithShape::Array(_) => saw_column = true,
            ArithShape::TVector(_) => saw_row = true,
            _ => {}
        }
    }
    if saw_column && saw_row {
        return Err(EmitError::at(
            id,
            format!(
                "`{head}`'s argument mixes vector orientations — a one-dimensional array \
                 (column) and a transposed vector (row) in one container. §03 \"Arrays\" gives \
                 an array a single element type and makes a transposed vector a distinct type \
                 from a one-dimensional array, so this is not a well-typed vector of vectors \
                 even though both lower to the same `tensor<nxf32>`. Transpose the odd element \
                 to match"
            ),
        ));
    }
    Ok(())
}

/// Lower §07 "Array and table operations"'s `addaxes(A, n_leading,
/// n_trailing)`, which "reshapes array `A` by adding `n_leading` singular
/// (size-one) axes before the axes of `A` and `n_trailing` singular axes after
/// them" — one `stablehlo.reshape`, since the element count is unchanged. §07's
/// own worked case is the check: `A` of size `(3, 4, 5)` with `addaxes(A, 2, 3)`
/// gives `(1, 1, 3, 4, 5, 1, 1, 1)`.
///
/// `addaxes(A, 0, 0)` emits nothing — the identity reshape would be noise.
///
/// The counts are read as LITERALS, not off the node's own inferred shape: §07
/// requires them to be "non-negative fixed integers", and a fixed expression is
/// already folded by the time the emitter runs (`addaxes(v, 1 + 1, 0)`
/// determinizes to `addaxes(v, 2, 0)`), so the literal path is both the exact
/// spec rule and self-contained — it does not depend on inference having
/// resolved the result shape, and it cannot silently disagree with it.
///
/// Refuses a TRANSPOSED-vector `A`: §07's domain column for this entry is
/// "array", not "vector", so §03's "the term vector will represent both …"
/// blanket does not widen it — and here the widening would change the answer
/// rather than merely permit a spelling. A row's tensor form is `[n]`, not
/// `[1, n]` (`crate::types::mlir_type_of` maps a `TVector{n}` and a rank-1
/// `Array[n]` to the same `tensor<nxf32>`), so `addaxes(transpose(v), 0, 1)`
/// would emit `[n, 1]` — a COLUMN — while the operand already reads as a row.
///
/// Refuses a dynamically-shaped or non-ranked `A` too, as [`lower_fill`] does
/// and for the same reason: `stablehlo.reshape`'s result shape is static text.
fn lower_addaxes(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a_id, nl_id, nt_id] = args_exact(id, args)?;
    let n_leading = axis_count(e, id, nl_id, "n_leading")?;
    let n_trailing = axis_count(e, id, nt_id, "n_trailing")?;
    if matches!(e.type_of(a_id), Some(Type::TVector { .. })) {
        return Err(EmitError::at(
            id,
            "addaxes: `A` is a transposed vector — §07 \"Array and table operations\" gives \
             `addaxes` the domain \"array, non-negative integer, non-negative integer\", and a \
             row's tensor form is `[n]`, so adding a trailing axis would produce a column \
             `[n, 1]` rather than the row the operand already reads as. Transpose it first",
        ));
    }
    let a = e.lower_node(a_id)?;
    let dims = match &a.ty {
        MlirTy::Ranked(dims) if dims.iter().all(Option::is_some) => dims.clone(),
        other => {
            return Err(EmitError::at(
                id,
                format!("addaxes: `A` must be a statically-shaped array, got {other:?}"),
            ));
        }
    };
    if n_leading == 0 && n_trailing == 0 {
        return Ok(a);
    }
    let mut out = vec![Some(1); n_leading];
    out.extend(dims);
    out.resize(out.len() + n_trailing, Some(1));
    Ok(e.reshape(&a, MlirTy::Ranked(out)))
}

/// One of `addaxes`' two axis counts, which §07 requires to be a "non-negative
/// fixed integer". Anything else refuses rather than being guessed — see
/// [`lower_addaxes`] for why the literal is the right source.
///
/// ONE message for both defects, because surface FlatPPL spells a negative count
/// as `neg(1)` rather than a negative literal: `addaxes(v, -1, 0)` reaches here
/// as a call, not a `Lit`, so a separate negative-literal arm would be a wording
/// no input can produce.
fn axis_count(e: &Emitter, id: NodeId, n_id: NodeId, which: &str) -> Result<usize, EmitError> {
    match e.node(e.resolve_ref_one(n_id)) {
        Node::Lit(Scalar::Int(n)) if *n >= 0 => Ok(*n as usize),
        _ => Err(EmitError::at(
            id,
            format!(
                "addaxes: `{which}` must be a non-negative fixed integer literal (§07 \"Array and \
                 table operations\"); a fixed arithmetic expression is folded before the emitter \
                 runs, so anything left here has no compile-time axis count"
            ),
        )),
    }
}

/// Which transposed-vector product [`lower_vector_product`] emits.
#[derive(Clone, Copy)]
enum VectorProduct {
    /// `transpose(a) * b` → scalar.
    Inner,
    /// `a * transpose(b)` → matrix.
    Outer,
    /// `transpose(a) * m` → row vector.
    RowMatrix,
}

/// Lower one of §07's two transposed-vector products. Both operands are rank-1
/// tensors (§03 makes the transposed vector a distinct type, not a distinct
/// tensor shape), so the ORIENTATION comes from the inferred types
/// [`classify_bare_mul`] already read — never from the lowered values, which
/// cannot tell a row from a column.
///
/// The inner product's lengths must agree; a mismatch refuses here rather than
/// reaching [`Emitter::inner_product`]'s panic. `infer`'s `mul_type` already
/// makes a statically-unequal pair `Type::Failed` ("inner product: vector lengths
/// disagree"), so the determiniser refuses first — this is the defensive second
/// line, matching [`lower_matrix_product`]'s own inner-dimension check. The outer
/// product needs no length agreement at all: `[n] × [m] → [n, m]` for any n, m.
///
/// [`VectorProduct::RowMatrix`] (`row[k] · [k, n] → row[n]`) is the third
/// orientation, and it runs ahead of the MERGED spec: §07's `mul` row lists
/// `matrix-vector` and no `vector-matrix` as of design `9e35262`, which is why the
/// preceding wave refused it. **flatppl-design#77 (pending owner review)** adds
/// `transposed-vector–matrix` to the row and states the result directly: "the
/// product of a transposed vector and a matrix is a transposed vector" — so a row,
/// not a single-row matrix, which is why the result type is `TVector{n}` and not a
/// `[1, n]` array. The maths agrees and would have forced it anyway:
/// `(1×k)(k×n) = 1×n`.
///
/// The MIRROR case, `matrix * row`, stays refused and is NOT the same omission:
/// `[m,k] · row[k]` does not conform for any `m, k` except the degenerate `k = 1`,
/// so there is no product to admit, and #77 does not add one.
fn lower_vector_product(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    kind: VectorProduct,
) -> Result<Value, EmitError> {
    let [a, b] = args_exact(id, args)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    let rank1 = |v: &Value| matches!(&v.ty, MlirTy::Ranked(d) if d.len() == 1);
    // The row–matrix case takes a rank-2 rhs; the two vector products take rank 1
    // on both sides.
    let want_rank2_rhs = matches!(kind, VectorProduct::RowMatrix);
    let rhs_ok = |v: &Value| matches!(&v.ty, MlirTy::Ranked(d) if d.len() == if want_rank2_rhs { 2 } else { 1 });
    if !rank1(&a) || !rhs_ok(&b) {
        return Err(EmitError::at(
            id,
            format!(
                "transposed-vector product needs a rank-1 lhs and a rank-{} rhs, got {:?} \
                 against {:?}",
                if want_rank2_rhs { 2 } else { 1 },
                a.ty,
                b.ty
            ),
        ));
    }
    match kind {
        VectorProduct::Inner => {
            let (MlirTy::Ranked(da), MlirTy::Ranked(db)) = (&a.ty, &b.ty) else {
                unreachable!("rank-1 checked above");
            };
            if matches!((da[0], db[0]), (Some(m), Some(n)) if m != n) {
                return Err(EmitError::at(
                    id,
                    format!(
                        "inner product operand lengths disagree: {:?} against {:?} — \
                         `transpose(a) * b` contracts the two vectors, so they must be the \
                         same length",
                        a.ty, b.ty
                    ),
                ));
            }
            Ok(e.inner_product(&a, &b))
        }
        VectorProduct::Outer => Ok(e.outer_product(&a, &b)),
        VectorProduct::RowMatrix => {
            let (MlirTy::Ranked(da), MlirTy::Ranked(db)) = (&a.ty, &b.ty) else {
                unreachable!("ranks checked above");
            };
            // The row's length pairs with the matrix's LEADING axis.
            if matches!((da[0], db[0]), (Some(k1), Some(k2)) if k1 != k2) {
                return Err(EmitError::at(
                    id,
                    format!(
                        "row-vector–matrix product inner dimensions disagree: {:?} against \
                         {:?} — `transpose(a) * m` contracts the row against the matrix's \
                         leading axis",
                        a.ty, b.ty
                    ),
                ));
            }
            Ok(e.row_matrix_product(&a, &b))
        }
    }
}

/// What a bare `add`/`sub`/`divide`/`pow` operand's inferred type says about its
/// §07 arithmetic shape. [`ArithShape::Unknown`] is the permissive class: it
/// covers `%deferred`, `%any`, an absent type (a freshly synthesized determiniser
/// node before re-inference) and every non-arithmetic type, so the guard below
/// refuses only a shape pair it can PROVE is outside the domain.
enum ArithShape {
    Scalar,
    /// A flat axis list, nested element chains flattened as
    /// `crate::types::mlir_type_of` flattens them. `None` is a
    /// dynamic axis, which matches any extent.
    Array(Vec<Option<u32>>),
    /// A transposed (row) vector — §03 keeps it distinct from a rank-1 `Array`,
    /// so the two are never "the same shape".
    TVector(Vec<Option<u32>>),
    Unknown,
}

impl ArithShape {
    fn is_array(&self) -> bool {
        matches!(self, ArithShape::Array(_) | ArithShape::TVector(_))
    }

    /// The pair is PROVABLY not the same shape. A dynamic axis against anything
    /// is not a proof, so it passes; only a statically-unequal extent, a rank
    /// difference, an array-against-scalar mix, or an `Array`/`TVector`
    /// transposition mismatch counts.
    fn differs_from(&self, other: &ArithShape) -> bool {
        use ArithShape::*;
        match (self, other) {
            (Unknown, _) | (_, Unknown) => false,
            (Scalar, Scalar) => false,
            (Scalar, _) | (_, Scalar) => true,
            // A column vector against a row vector.
            (Array(_), TVector(_)) | (TVector(_), Array(_)) => true,
            (Array(a), Array(b)) | (TVector(a), TVector(b)) => {
                a.len() != b.len()
                    || a.iter()
                        .zip(b.iter())
                        .any(|(x, y)| matches!((x, y), (Some(m), Some(n)) if m != n))
            }
        }
    }
}

/// Classify `id`'s inferred type for the bare-arithmetic domain guard.
fn arith_shape(e: &Emitter, id: NodeId) -> ArithShape {
    /// The axes an `Array`/`TVector` element chain contributes, or `None` for a
    /// scalar leaf.
    fn axes(ty: &Type) -> Option<Vec<Option<u32>>> {
        let dim = |d: &flatppl_core::Dim| match d {
            flatppl_core::Dim::Static(n) => Some(*n),
            flatppl_core::Dim::Dynamic => None,
        };
        match ty {
            Type::Array { shape, elem } => {
                let mut v: Vec<Option<u32>> = shape.iter().map(dim).collect();
                v.extend(axes(elem).unwrap_or_default());
                Some(v)
            }
            Type::TVector { len, elem } => {
                let mut v = vec![dim(len)];
                v.extend(axes(elem).unwrap_or_default());
                Some(v)
            }
            _ => None,
        }
    }
    match e.type_of(id) {
        Some(Type::Scalar(_)) => ArithShape::Scalar,
        Some(ty @ Type::Array { .. }) => ArithShape::Array(axes(ty).unwrap_or_default()),
        Some(ty @ Type::TVector { .. }) => ArithShape::TVector(axes(ty).unwrap_or_default()),
        _ => ArithShape::Unknown,
    }
}

/// Lower a bare `add`/`sub`/`divide`/`pow` — the surface `+`, `-`, `/`, `^` —
/// refusing an operand pair outside the §07 "Operator-equivalent functions"
/// domain for that head:
///
/// - `add`/`sub`: "scalars or arrays of same shape (real or complex)". A scalar
///   against an array is OUTSIDE it, so `scalar + vector` refuses; two arrays of
///   the same shape are vector addition and stay legal.
/// - `divide`: "scalars, array-scalar, transposed-vector–scalar (real or complex)"
///   (flatppl-design#77, pending owner review; it supersedes the narrower row #75
///   introduced). §05 "No implicit operator broadcasting" states the same
///   constraint directly — "`/` requires a scalar divisor, and `^` is
///   scalar-only" — so the DIVISOR is the whole discriminator: a dividend of ANY
///   rank over a scalar divisor is scalar multiplication by the reciprocal, sound,
///   and lowers as the ordinary scalar-broadcast divide. `array` is §03's
///   rank-agnostic n-dimensional term, and `transposed-vector–scalar` names the
///   row-vector dividend outright. `scalar / vector` is NOT in the domain, and
///   neither is `array / array`; an elementwise reciprocal is `./`'s job.
/// - `pow`: "scalars (real or complex)". Any array operand refuses.
///
/// `neg` needs no guard: its domain is "scalars or arrays", so elementwise
/// negation of an array is already sound.
///
/// Reached ONLY from `Emitter::lower_node`'s dispatch, never from
/// `Emitter::lower_broadcast` — the dotted spellings (`.+`, `.-`, `./`, `.^`)
/// arrive as `broadcast(add, …)` etc. and re-enter [`lower_builtin`] directly, so
/// they keep broadcasting. Same discriminator as [`lower_bare_mul`]'s.
///
/// Classifying on the INFERRED types (not the lowered shapes) is what keeps the
/// determiniser's synthesized arithmetic lowering: a node it built fresh has no
/// type yet, and a `%local` inside a `functionof` monomorphised under `broadcast`
/// is typed scalar even though its bound value is rank-1.
pub(crate) fn lower_bare_arith<'m>(
    e: &mut Emitter<'m>,
    id: NodeId,
    head: &str,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    type BinOp<'m> = fn(&mut Emitter<'m>, &Value, &Value) -> Value;
    let (op, surface, dotted): (BinOp<'m>, &str, &str) = match head {
        "add" => (Emitter::add, "+", ".+"),
        "sub" => (Emitter::sub, "-", ".-"),
        "divide" => (Emitter::div, "/", "./"),
        "pow" => (Emitter::pow, "^", ".^"),
        other => {
            return Err(EmitError::at(
                id,
                format!("not a bare arithmetic head '{other}'"),
            ));
        }
    };
    // Wrong arity: let `binary`'s `args_exact` produce the arity message.
    let Ok([a, b]) = <[NodeId; 2]>::try_from(args) else {
        return binary(e, id, args, op);
    };
    let (sa, sb) = (arith_shape(e, a), arith_shape(e, b));
    let shape = |n: NodeId| match e.type_of(n) {
        Some(t) => format!("{t:?}"),
        None => "unknown".to_string(),
    };
    let refuse = |domain: &str| {
        Err(EmitError::at(
            id,
            format!(
                "`{surface}` has no meaning for these operand shapes: {} against {} — §07 \
                 \"Operator-equivalent functions\" gives `{head}` the domain {domain}. Write \
                 `{dotted}` for the elementwise form (`broadcast({head}, …)`), which broadcasts \
                 a scalar against an array",
                shape(a),
                shape(b)
            ),
        ))
    };
    // The domain named in the refusal a USER sees. It must track the row the guard
    // actually enforces: quoting the narrower pre-#77 row told anyone hitting the
    // divisor refusal that a dividend shape the engine accepts was out of bounds.
    const DIVIDE_DOMAIN: &str = "\"scalars, array-scalar, transposed-vector–scalar\"";
    match head {
        "pow" if sa.is_array() || sb.is_array() => refuse("\"scalars\""),
        // A known-array DIVISOR is proof enough on its own, whatever the dividend
        // is: it rules out `scalar / vector` and `vector / vector` together.
        "divide" if sb.is_array() => refuse(DIVIDE_DOMAIN),
        // Any-rank dividend over a scalar divisor lowers. The DIVISOR is the whole
        // discriminator: §05 "No implicit operator broadcasting" constrains only it
        // ("`/` requires a scalar divisor", unchanged by #77), and `v / s` is scalar
        // multiplication by the reciprocal, which is sound at every rank.
        //
        // Ahead of the MERGED spec row for rank 3 and above: §07's row enumerates
        // `vector-scalar` and `matrix-scalar` and stops (design `9e35262`), which is
        // why the preceding wave refused a rank-3 dividend. **flatppl-design#77
        // (pending owner review)** rewrites the row as "scalars, array-scalar,
        // transposed-vector–scalar", where `array` is §03's rank-agnostic
        // n-dimensional term — so every rank is admitted, and the transposed-vector
        // dividend is now named OUTRIGHT rather than resting on §03's "the term
        // vector will represent both" blanket, which is what the F6 fix argued from.
        // Nothing further is needed here: every dividend shape falls through, so the
        // only `divide` refusal left is the array-divisor one above.
        "add" | "sub" if sa.differs_from(&sb) => refuse("\"scalars or arrays of same shape\""),
        _ => binary(e, id, args, op),
    }
}

/// Lower a bare `mul` [`classify_bare_mul`] called a matrix product: matrix·
/// vector via [`Emitter::matvec`], matrix·matrix via [`Emitter::matmat`], both
/// one `stablehlo.dot_general`. The operand RANKS and the inner dimensions are
/// re-checked HERE so a disagreement refuses; the emitter helpers panic on one,
/// and this is the only caller that reaches them from surface `*`.
fn lower_matrix_product(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a, b] = args_exact(id, args)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    let (MlirTy::Ranked(da), MlirTy::Ranked(db)) = (&a.ty, &b.ty) else {
        return Err(EmitError::at(
            id,
            "matrix product needs ranked tensor operands",
        ));
    };
    // The ranks come from the INFERRED types (`classify_bare_mul`) while the dims
    // below come from the LOWERED values; if those two ever disagree, indexing
    // `da[1]`/`db[0]` would panic in the one function whose job is to refuse
    // before the emitter's own assertions are reached. Check rather than index
    // blindly.
    if da.len() != 2 || db.is_empty() || db.len() > 2 {
        return Err(EmitError::at(
            id,
            format!(
                "matrix product operand ranks disagree with their inferred types: \
                 {:?} against {:?} — expected a rank-2 lhs and a rank-1 or rank-2 rhs",
                a.ty, b.ty
            ),
        ));
    }
    // A dynamic inner dim on either side is not a KNOWN disagreement, so it
    // passes: `dot_general` contracts it at run time.
    if matches!((da[1], db[0]), (Some(m), Some(n)) if m != n) {
        return Err(EmitError::at(
            id,
            format!(
                "matrix product inner dimensions disagree: {:?} against {:?} — the lhs's \
                 trailing axis must match the rhs's leading axis",
                a.ty, b.ty
            ),
        ));
    }
    if db.len() == 1 {
        Ok(e.matvec(&a, &b))
    } else {
        Ok(e.matmat(&a, &b))
    }
}

// ---- reductions --------------------------------------------------------------

/// Lower `sum`. Over an array this is the ordinary reduction; over a TABLE it
/// refuses, and refuses SPECIFICALLY rather than incidentally.
///
/// §07 "Table reductions" makes `sum(t)` return "a record whose fields are the
/// column names and values are the per-column reductions" — and this emitter has no
/// record value at all: every [`Value`] is a tensor, and [`lower_builtin`]'s
/// `"record"` arm refuses with "record has no tensor form". So the result is not
/// expressible here, and refusing is the honest outcome rather than lowering one
/// column and calling it the answer.
///
/// Checked BEFORE the argument is lowered, because otherwise the argument's own
/// refusal fires first and blames the wrong thing: a table from `load_data` gave the
/// column-wise-destructuring message (accurate, but about `load_data` rather than
/// about the reduction), and a table from `elementof` gave "unsupported builtin head
/// 'elementof'", which says nothing about tables at all. Both are this message now,
/// whatever the table's provenance.
///
/// `mean`/`var`/`std` need no equivalent: this map lowers them for NO argument type,
/// so "unsupported builtin head" is already accurate for them.
fn lower_sum(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    if let [arg] = args {
        if matches!(e.type_of(*arg), Some(Type::Table { .. })) {
            return Err(EmitError::at(
                id,
                "a table reduction has no tensor form: §07 \"Table reductions\" makes `sum` over \
                 a table a RECORD of per-column sums, and this emitter represents every value as \
                 a tensor. Reduce one column at a time instead — `sum(data.x)` lowers, and a \
                 table input is already one argument per column",
            ));
        }
    }
    unary(e, id, args, Emitter::reduce_sum)
}

// ---- ifelse / inf -----------------------------------------------------------

fn lower_ifelse(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [c, a, b] = args_exact(id, args)?;
    require_predicate_head(e, c, "ifelse condition")?;
    // The two BRANCHES must share an orientation: selecting between a row and a
    // column is not defined, and neither the `broadcast_pair` below nor
    // `Emitter::select` can see the difference (both are `tensor<nxf32>`).
    require_same_orientation(e, id, a, b)?;
    let c = e.lower_node(c)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    // `Emitter::select` broadcasts the two BRANCHES against each other through
    // `broadcast_pair` (which panics on a pair with no broadcast form), and then
    // broadcasts both against the first ranked shape among {pred, a, b}. So two
    // pairings have to hold, and neither is checked by the infallible helper:
    require_broadcastable(id, &a, &b)?;
    require_select_predicate(id, &c, &a, &b)?;
    Ok(e.select(&c, &a, &b))
}

/// Refuse a `select` predicate whose shape `Emitter::select`'s second pass cannot
/// legally broadcast the branches to.
///
/// That pass is NOT `broadcast_pair` — it is `Emitter::broadcast_scalar`, which
/// emits `broadcast_in_dim(s, &[], out_ty)` whenever `s.ty != out_ty`, and an
/// EMPTY `dims` list is valid only for a rank-0 operand. So the size-1 expansion
/// [`require_broadcastable`] permits is wrong here: it is sound for elementwise
/// arithmetic, where `broadcast_pair` supplies proper identity dims, but a size-1
/// predicate against `[3]` branches (or the mirror) silently emitted
/// `broadcast_in_dim … dims = [] : (tensor<3xf32>) -> tensor<1xf32>` — invalid
/// StableHLO, exit 0.
///
/// The admissible cases, and nothing else:
/// - a SCALAR predicate — StableHLO's `select` takes one against ranked operands,
///   and the branches keep their own shape;
/// - a ranked predicate with SCALAR branches — they broadcast up from rank 0, so
///   the empty `dims` list is correct;
/// - a ranked predicate whose shape EQUALS the shape the branches reconcile to,
///   making the second pass a no-op.
///
/// The last case compares against the RECONCILED branch shape, not against `a`
/// alone: `broadcast_pair` has already expanded a size-1 branch axis with proper
/// dims by the time the predicate pass runs, so `(pred [3], a [1], b [3])` is
/// legal and emits a valid `select` — checking `c == a` would refuse it.
///
/// This checks SHAPE only, never orientation: a predicate derived from transposed
/// vectors selecting between plain ones is deliberately admitted. `select` is
/// per-element and the result takes the BRANCHES' orientation (which
/// [`require_same_orientation`] does check), so the mask's orientation has no
/// channel through which to change a selected value or the result type — unlike
/// an elementwise `add` or a `compare`, where both operands feed the result.
fn require_select_predicate(id: NodeId, c: &Value, a: &Value, b: &Value) -> Result<(), EmitError> {
    if matches!(c.ty, MlirTy::Scalar) {
        return Ok(());
    }
    let MlirTy::Ranked(dc) = &c.ty else {
        return Err(EmitError::at(
            id,
            format!("ifelse predicate has no tensor form: {:?}", c.ty),
        ));
    };
    // The shape the branches carry into the predicate pass. `require_broadcastable`
    // has already run on the pair, so a mixed Ranked pair here is compatible.
    let branch_shape = match (&a.ty, &b.ty) {
        (MlirTy::Ranked(da), MlirTy::Ranked(db)) => Some(common_shape(da, db)),
        (MlirTy::Ranked(d), _) | (_, MlirTy::Ranked(d)) => Some(d.clone()),
        // Both scalar: they broadcast up from rank 0, which `dims = []` expresses
        // correctly whatever the predicate's shape.
        _ => None,
    };
    match branch_shape {
        None => Ok(()),
        Some(target) if *dc == target => Ok(()),
        Some(target) => Err(EmitError::at(
            id,
            format!(
                "ifelse predicate shape {:?} does not match its branches' shape {:?} — a \
                 select predicate must be a scalar or exactly the branch shape (the branch \
                 broadcast cannot expand a non-scalar predicate)",
                MlirTy::Ranked(dc.clone()),
                MlirTy::Ranked(target)
            ),
        )),
    }
}

/// The per-axis shape `Emitter::broadcast_pair` reconciles a COMPATIBLE ranked
/// pair to: an axis pair that is equal keeps its size, a size-1 axis takes the
/// other side's, and a doubly dynamic axis stays dynamic. Only meaningful after
/// [`require_broadcastable`] has accepted the pair.
fn common_shape(da: &[Option<u64>], db: &[Option<u64>]) -> Vec<Option<u64>> {
    da.iter()
        .zip(db.iter())
        .map(|(&x, &y)| match (x, y) {
            (Some(1), other) => other,
            (other, Some(1)) => other,
            (Some(m), _) => Some(m),
            (None, other) => other,
        })
        .collect()
}

/// The predicate-producing builtin heads this map lowers to an `i1` value.
/// [`Emitter::select`] and [`Emitter::and`] unconditionally render their
/// predicate operands as `i1`, so handing either any other node (e.g. a bare
/// `Lit(Bool)`, which lowers as a plain `tensor<f32>` `dense<1.0>` via
/// `constant`) would make the declared `i1` operand disagree with the actual
/// emitted type, producing ill-typed StableHLO.
/// The heads whose CALL NODE this map recognizes as a boolean predicate — the
/// operand vocabulary of `ifelse`'s condition and of §07's logical connectives.
///
/// Every entry lowers to an `i1`-typed [`Value`], so the list is exactly "the
/// boolean-producing heads this map lowers". It does NOT admit a `Bool`-typed
/// VALUE (a bound boolean, a boolean ABI input), which stays refused as a
/// separate gap (`flatppl-dev/stablehlo-feature-matrix.md`, prioritized gap 6).
const PREDICATE_HEADS: &[&str] = &[
    "in", "compare", "lt", "gt", "le", "ge", "land", "lor", "lxor", "lnot", "iszero", "equal",
    "unequal", "isfinite", "isinf", "isnan",
];

/// An `ifelse` condition / `land` operand must be one of
/// [`PREDICATE_HEADS`]. Same narrow-and-refuse discipline as `get`/`get0`'s
/// literal-selector check: checked structurally against the *unlowered* node,
/// before `lower_node` ever runs on it.
fn require_predicate_head(e: &Emitter, cond: NodeId, what: &str) -> Result<(), EmitError> {
    let is_predicate = matches!(
        e.node(cond),
        Node::Call(c) if matches!(
            c.head,
            CallHead::Builtin(sym) if PREDICATE_HEADS.contains(&e.resolve(sym))
        )
    );
    if is_predicate {
        Ok(())
    } else {
        Err(EmitError::at(
            cond,
            format!(
                "{what} must be a boolean predicate ({})",
                PREDICATE_HEADS.join("/")
            ),
        ))
    }
}

// ---- comparisons and logic ----------------------------------------------------

/// §07 `lt`/`gt` — one `stablehlo.compare` in direction `dir`. Operand shape
/// and elem-kind reconciliation is [`Emitter::compare`]'s.
fn lower_compare(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    dir: &str,
) -> Result<Value, EmitError> {
    let [a, b] = args_exact(id, args)?;
    // Comparing a row against a column is not defined either, and `MlirTy` cannot
    // tell them apart — same reasoning as [`binary`]'s.
    require_same_orientation(e, id, a, b)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    // `Emitter::compare` reconciles its operands through `broadcast_pair` exactly
    // as `Emitter::binary` does, and is equally infallible.
    require_broadcastable(id, &a, &b)?;
    Ok(e.compare(dir, &a, &b))
}

/// §07 "Comparison functions" `equal`/`unequal` — one `stablehlo.compare`
/// `EQ`/`NE`.
///
/// The operand domain is §07's verbatim: "`integers`, `booleans`, strings", and
/// §07 states why — "Exact equality (`equal` / `==` and `unequal` / `!=`) is
/// restricted to discrete domains to avoid dependence on numerical precision.
/// To compare real-valued quantities for exact equality, use a function that
/// guarantees a discrete result like `integer(x)`, `floor(x)`, `ceil(x)`, or
/// `round(x)`." So a `Real` operand is REFUSED rather than compared: emitting
/// a float `compare EQ` would answer a question §07 declines to define, and
/// §07 already names `iszero` as the one exact test that does admit a
/// non-discrete input. A string operand has no tensor form and refuses in
/// `crate::types` before reaching here.
///
/// The refusal reads the LOWERED operand kinds, not the inferred types: a
/// determiniser-synthesized node may carry no type, and it is the emitted
/// element type that decides whether the comparison is float or integer.
fn lower_exact_compare(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    dir: &str,
) -> Result<Value, EmitError> {
    let [a_id, b_id] = args_exact(id, args)?;
    require_same_orientation(e, id, a_id, b_id)?;
    let a = e.lower_node(a_id)?;
    let b = e.lower_node(b_id)?;
    require_broadcastable(id, &a, &b)?;
    for v in [&a, &b] {
        if v.elem == ElemKind::Real {
            return Err(EmitError::at(
                id,
                "equal/unequal: §07 \"Comparison functions\" gives these the domain \
                 \"`integers`, `booleans`, strings\" and states that exact equality \"is \
                 restricted to discrete domains to avoid dependence on numerical precision\". \
                 A real operand has no `==` lowering — wrap it in a function that guarantees \
                 a discrete result (`integer`, `floor`, `ceil`, `round`), or use `iszero`, \
                 which §07 defines for a non-discrete input",
            ));
        }
    }
    Ok(e.compare(dir, &a, &b))
}

/// Which §07 "Logical operators" connective a two-operand logic head is.
#[derive(Clone, Copy)]
enum Connective {
    /// `land` — `a && b`.
    And,
    /// `lor` — `a || b`.
    Or,
    /// `lxor` — §07's exclusive disjunction ("no infix operator").
    Xor,
}

/// §07 `land`/`lor`/`lxor` (`a && b`, `a || b`, exclusive or, all over
/// `booleans`) — one `stablehlo.and`/`or`/`xor` over two `i1` predicates.
///
/// Both operands must be [`PREDICATE_HEADS`] calls, and must share a shape:
/// [`Emitter::and`] and its siblings render ONE type for both operands and the
/// result, so a mismatched pair would emit ill-typed text.
///
/// The [`PREDICATE_HEADS`] requirement is NOT "must be boolean" — a `Bool`-typed
/// bound value or ABI input renders `i1` too and would emit fine. It is the
/// deliberately narrow same-shape gate `ifelse` uses, kept identical here so the
/// boolean-VALUE gap is one documented refusal rather than half-closed in two
/// inconsistent places (`flatppl-dev/stablehlo-feature-matrix.md`, prioritized
/// gap 6).
fn lower_land(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Connective,
) -> Result<Value, EmitError> {
    let name = match which {
        Connective::And => "land",
        Connective::Or => "lor",
        Connective::Xor => "lxor",
    };
    let [a_id, b_id] = args_exact(id, args)?;
    require_predicate_head(e, a_id, &format!("{name} operand"))?;
    require_predicate_head(e, b_id, &format!("{name} operand"))?;
    let a = e.lower_node(a_id)?;
    let b = e.lower_node(b_id)?;
    if a.ty != b.ty {
        return Err(EmitError::at(
            id,
            format!(
                "{name}: operands must have the same shape, got {:?} and {:?}",
                a.ty, b.ty
            ),
        ));
    }
    Ok(match which {
        Connective::And => e.and(&a, &b),
        Connective::Or => e.or(&a, &b),
        Connective::Xor => e.xor(&a, &b),
    })
}

/// §07 `lnot` (`!a`, over `booleans`) — one `stablehlo.not`. Same
/// [`PREDICATE_HEADS`] operand rule as [`lower_land`], for the same reason.
fn lower_lnot(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a_id] = args_exact(id, args)?;
    require_predicate_head(e, a_id, "lnot operand")?;
    let a = e.lower_node(a_id)?;
    Ok(e.not(&a))
}

/// Which §07 "Scalar predicates" finiteness test a head is.
#[derive(Clone, Copy)]
enum Finiteness {
    /// `isfinite` — "`x` is a finite number (not ±∞, not NaN)".
    Finite,
    /// `isinf` — "`x` is $+\infty$ or $-\infty$".
    Inf,
    /// `isnan` — "`x` is NaN".
    Nan,
}

/// §07 `isfinite`/`isinf`/`isnan`, composed out of ops this crate has already
/// validated rather than the `chlo.is_inf` family:
///
/// - `isnan(x)` is `x != x`, the IEEE-754 definition (NaN is the one value not
///   equal to itself);
/// - `isinf(x)` is `abs(x) == inf`, true for both signs and false for NaN
///   (`NaN == inf` is false);
/// - `isfinite(x)` is `abs(x) < inf`, false for ±∞ AND false for NaN (an
///   unordered comparison is false), which is exactly §07's "not ±∞, not NaN".
///
/// An `Int` or `Bool` operand takes the same path: [`Emitter::compare`] widens
/// it to the float dtype exactly (§03 `booleans ⊂ integers ⊂ reals`), where
/// `isfinite` is `true` and the other two `false` — the right answers for a
/// value family with no infinities.
fn lower_finiteness(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Finiteness,
) -> Result<Value, EmitError> {
    let [a_id] = args_exact(id, args)?;
    let a = e.lower_node(a_id)?;
    if matches!(which, Finiteness::Nan) {
        return Ok(e.compare("NE", &a, &a));
    }
    let mag = e.abs(&a);
    let inf = e.inf(mag.ty.clone());
    Ok(match which {
        Finiteness::Finite => e.compare("LT", &mag, &inf),
        Finiteness::Inf => e.compare("EQ", &mag, &inf),
        Finiteness::Nan => unreachable!("handled above"),
    })
}

/// §07 `iszero` — `stablehlo.compare EQ` against zero. §07: "`iszero` checks
/// that its argument is exactly zero, with no tolerance for numerical
/// precision", so NO epsilon is introduced. An `Int` operand is widened to the
/// float dtype by [`Emitter::compare`]'s own kind reconciliation, which is
/// exact (§03 `integers ⊂ reals`) and so preserves exactness.
fn lower_iszero(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    let a = e.lower_node(a)?;
    let zero = e.constant(0.0, a.ty.clone());
    Ok(e.compare("EQ", &a, &zero))
}

// ---- real / extrema / fill -----------------------------------------------------

/// §07 `real` — "returns `x` for real `x`". Value identity: NO rounding, clamping or
/// truncation. It may emit the exact §03 `booleans ⊂ integers ⊂ reals` embedding for an
/// integer-typed operand, and nothing for an already-real one. Decided on the operand's
/// own lowered kind, not the node's name — the determiniser wraps `round` in `real` to
/// stop `exp(round(x))` typing as `integers` and evaluating in integer arithmetic.
///
/// A `Complex` operand (where §07 gives $\mathrm{Re}(x)$) has no tensor form
/// at all and is refused upstream in `crate::types`.
fn lower_real(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    let a = e.lower_node(a)?;
    Ok(e.convert(&a, ElemKind::Real))
}

#[derive(Clone, Copy)]
enum Extremum {
    Max,
    Min,
}

/// §07 `maximum(xs)` / `minimum(xs)` — a full reduction over a real ARRAY
/// ($\max_i x_i$ / $\min_i x_i$). A scalar operand is refused rather than
/// silently returned: §07 lists these under reductions with domain "real
/// arrays", and the binary §07 `max`/`min` are different functions this map
/// does not lower. A non-`Real` operand is refused too — the ±inf reduction
/// identity has no integer or boolean form (see [`Emitter::reduce_min`]).
fn lower_extremum(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Extremum,
) -> Result<Value, EmitError> {
    let [xs] = args_exact(id, args)?;
    let xs = e.lower_node(xs)?;
    if !matches!(xs.ty, MlirTy::Ranked(_)) {
        return Err(EmitError::at(
            id,
            format!(
                "maximum/minimum: operand must be an array, got {:?} (the binary \
                 max/min are separate functions with no lowering here)",
                xs.ty
            ),
        ));
    }
    if xs.elem != ElemKind::Real {
        return Err(EmitError::at(
            id,
            format!(
                "maximum/minimum: only a real array is supported, got {:?}",
                xs.elem
            ),
        ));
    }
    Ok(match which {
        Extremum::Max => e.reduce_max(&xs),
        Extremum::Min => e.reduce_min(&xs),
    })
}

/// §07 "Elementary functions" `min(a, b)` / `max(a, b)` — the BINARY pair,
/// one `stablehlo.minimum`/`maximum`. Distinct from [`lower_extremum`]'s
/// same-family `minimum`/`maximum` reductions, which take one array.
///
/// A `Bool` operand refuses: §07's domain here is `reals`, and while §03 nests
/// `booleans ⊂ integers ⊂ reals` so an INTEGER operand is in domain (and keeps
/// an integer result through [`Emitter::min`]'s kind-polymorphic
/// [`Emitter::binary`]), `stablehlo.minimum` over `i1` is a conjunction, which
/// is §07's `land` rather than its `min`.
fn lower_binary_extremum(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Extremum,
) -> Result<Value, EmitError> {
    let [a_id, b_id] = args_exact(id, args)?;
    require_same_orientation(e, id, a_id, b_id)?;
    let a = e.lower_node(a_id)?;
    let b = e.lower_node(b_id)?;
    require_broadcastable(id, &a, &b)?;
    for v in [&a, &b] {
        if v.elem == ElemKind::Bool {
            return Err(EmitError::at(
                id,
                "min/max: §07 \"Elementary functions\" gives the binary `min`/`max` the domain \
                 `reals`, and over booleans `stablehlo.minimum`/`maximum` is a conjunction / \
                 disjunction — §07's `land`/`lor`, not its `min`/`max`. Convert to reals first",
            ));
        }
    }
    Ok(match which {
        Extremum::Max => e.max(&a, &b),
        Extremum::Min => e.min(&a, &b),
    })
}

/// §07 `fill(x, size)` — "creates an array of shape `size` filled with value
/// `x`", one `stablehlo.broadcast_in_dim` of the scalar `x`.
///
/// The result shape is read off `id`'s OWN inferred type, not by lowering
/// `size`: the determiniser spells the size as `lengthof(v)`, which has no
/// tensor form, while inference has already resolved the result shape. A
/// dynamic (`?`) axis is refused — `broadcast_in_dim`'s result shape must be
/// static text.
///
/// A fill value whose kind OUTRANKS the array's element kind is refused too:
/// [`Emitter::convert`] is exact only going UP §03's `booleans ⊂ integers ⊂ reals`
/// chain, and a real value into an integer array would truncate toward zero.
/// Inference should reject that upstream, so this is a narrow-and-refuse guard.
fn lower_fill(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [x_id, _size] = args_exact(id, args)?;
    let (ty, kind) = e.node_ty(id)?;
    match &ty {
        MlirTy::Ranked(dims) if dims.iter().all(Option::is_some) => {}
        other => {
            return Err(EmitError::at(
                id,
                format!("fill: result must be a statically-shaped array, got {other:?}"),
            ));
        }
    }
    let x = e.lower_node(x_id)?;
    if x.ty != MlirTy::Scalar {
        return Err(EmitError::at(
            id,
            format!("fill: fill value must be a scalar, got {:?}", x.ty),
        ));
    }
    if elem_rank(x.elem) > elem_rank(kind) {
        return Err(EmitError::at(
            id,
            format!(
                "fill: fill value is {:?} but the array element type is {kind:?}; \
                 narrowing it would not be value-preserving",
                x.elem
            ),
        ));
    }
    let x = e.convert(&x, kind);
    Ok(e.broadcast_in_dim(&x, &[], ty))
}

fn lower_inf(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    args_exact::<0>(id, args)?;
    Ok(e.inf(MlirTy::Scalar))
}

/// §03 `pi` — "the mathematical constant $\pi$". Gate vocabulary: `atan`'s
/// image endpoint is `pi / 2` (`crate`-external, `determinizer::invert`).
/// `f64`'s `Display` is shortest-round-trip, so the emitted decimal literal
/// recovers the `f64` value exactly and the nearest `f32` to it at `Dtype::F32`
/// — the same rounding every other real literal in an f32 module takes.
fn lower_pi(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    args_exact::<0>(id, args)?;
    Ok(e.scalar(std::f64::consts::PI))
}

// ---- logsumexp ---------------------------------------------------------------

/// `logsumexp(v)` (spec §07) via the numerically-stable shift-by-max
/// identity: `log(Σ exp(v - max(v))) + max(v)`. The determiniser always
/// wraps its argument in a `vector(t1, …, tk)` call (superpose/discrete-
/// marginal); `lower_node`'s `"vector"` head (below, [`Emitter::vector`])
/// is what turns that into the rank-1 tensor `v` this function reduces
/// over — this function itself only ever sees the one already-resolved
/// argument node, whatever built it. `max(v)`/`Σ` reduce to a `Scalar`;
/// `v - max(v)` needs `max(v)` broadcast back up to `v`'s shape first
/// (StableHLO's elementwise ops require identical operand shapes — no
/// implicit scalar broadcast).
fn lower_logsumexp(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [v] = args_exact(id, args)?;
    let v = e.lower_node(v)?;
    let m = e.reduce_max(&v);
    let m_bc = broadcast_to(e, id, &m, &v.ty)?;
    let shifted = e.sub(&v, &m_bc);
    let exp_shifted = e.exp(&shifted);
    let sum = e.reduce_sum(&exp_shifted);
    let log_sum = e.log(&sum);
    Ok(e.add(&log_sum, &m))
}

/// Broadcast a value `a` up to `ty`'s shape via [`Emitter::broadcast_in_dim`]
/// when the shapes differ; returns `a` unchanged (no op emitted) when they
/// already match — e.g. `logsumexp` over a length-1 vector, or `in`'s bound
/// already matching a scalar variate. Two shape-mismatched forms are
/// supported, both under an IDENTITY dimension mapping (spec §04
/// "Broadcasting" — size-1 axes expand by repetition, never NumPy-style
/// rank-prepending):
///
/// - `a` is a `Scalar`: the established rank-0 `dims = []` broadcast form
///   (`logsumexp`'s reduced max, `in`'s interval bounds — both callers'
///   existing case, unchanged);
/// - `a` is `Ranked` with the SAME RANK as `ty` and every axis either
///   already matches `ty`'s or is size-1 (a length-1 `iid(Dist, n)`
///   parameter broadcasting up to the length-`n` variate's shape, reached
///   here if a future caller passes a ranked size-1 operand — today's two
///   callers only ever pass a `Scalar` `a`, so this arm is not yet
///   exercised but is no longer refused).
///
/// Refuses (rather than mis-emitting a shape-mismatched op) for anything
/// else — a rank mismatch, or an axis that is neither equal nor size-1.
fn broadcast_to(e: &mut Emitter, id: NodeId, a: &Value, ty: &MlirTy) -> Result<Value, EmitError> {
    if &a.ty == ty {
        return Ok(a.clone());
    }
    match (&a.ty, ty) {
        (MlirTy::Scalar, _) => Ok(e.broadcast_in_dim(a, &[], ty.clone())),
        (MlirTy::Ranked(da), MlirTy::Ranked(db))
            if da.len() == db.len() && da.iter().zip(db).all(|(x, y)| x == y || *x == Some(1)) =>
        {
            let dims: Vec<u64> = (0..da.len() as u64).collect();
            Ok(e.broadcast_in_dim(a, &dims, ty.clone()))
        }
        _ => Err(EmitError::at(
            id,
            format!("shape mismatch: cannot broadcast {:?} to {ty:?}", a.ty),
        )),
    }
}

// ---- vector -------------------------------------------------------------------

/// `vector(t1, …, tk)` (spec §07 vector literal): packs `k` already-lowered
/// elements into a tensor one rank higher than the elements via
/// [`Emitter::vector`] — scalar elements (the determiniser's own shape,
/// wrapping a `logsumexp` argument for superpose/discrete-marginal) stack
/// into a rank-1 tensor; same-shape ARRAY elements (a legal
/// vector-of-vectors, spec §03 — distinct from a matrix) stack into a
/// rank-2-or-higher tensor. Refuses on zero elements (`concatenate` needs at
/// least one operand, and `Emitter::vector` asserts on that as an internal
/// invariant, not a well-formed-but-empty case worth tolerating here) and on
/// RAGGED elements (not all the same `MlirTy` — e.g. vector-of-vectors whose
/// inner vectors have different lengths): §03 arrays are fixed-size/
/// rectangular, so a ragged `vector(...)` has no tensor form at all —
/// refused here, before `Emitter::vector`, rather than let its own
/// identical-shape assertion fire as an internal-invariant panic. Each
/// element is [`Emitter::convert`]ed to `id`'s own inferred elem kind first
/// (e.g. a homogeneous-`Int` literal array like `[2, 3, 5]` stays `Int`
/// throughout; a mixed literal array — individually-tagged `Lit` nodes that
/// inference has already unified to one array element type — converges on
/// that unified kind) so `Emitter::vector`'s own elem-uniformity invariant
/// always holds by construction, never by luck.
fn lower_vector(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    if args.is_empty() {
        return Err(EmitError::at(id, "vector: expected at least one element"));
    }
    let elems: Vec<Value> = args
        .iter()
        .map(|&a| e.lower_node(a))
        .collect::<Result<_, _>>()?;
    let elem_ty = &elems[0].ty;
    if elems.iter().any(|v| &v.ty != elem_ty) {
        return Err(EmitError::at(
            id,
            "vector elements must have identical shape; ragged vector-of-vectors has no tensor form",
        ));
    }
    let target = e.node_kind(id);
    let elems: Vec<Value> = elems.iter().map(|v| e.convert(v, target)).collect();
    Ok(e.vector(&elems))
}

// ---- get / get0 ---------------------------------------------------------------

/// `get0(container, index)` / `get(container, index)` (spec §07): zero- vs
/// one-based element access. Two cases are implemented:
///
/// - A **literal-integer** selector into a rank-1 tensor container (the
///   shape the determiniser itself emits) — [`lower_get_literal`], via
///   `slice` (extract the one element) + `reshape` (drop the now-length-1
///   axis, yielding a `Scalar`).
/// - A **runtime rank-1 `Int`-tensor** selector into a rank-1 tensor
///   container (the `theta[person]`-style vector-index case) —
///   [`lower_get_gather`], via [`Emitter::gather`].
///
/// Multi-selector / named-field / `all`/`only` forms (record, table, tuple),
/// multi-dimensional array access, and a non-`Int` runtime index (spec §07)
/// are refused, not guessed: `get`/`get0` can also reach this map from
/// user-authored FlatPDL, not just the determiniser's own output, and none
/// of those forms has an obvious single-op tensor lowering.
fn lower_get(e: &mut Emitter, id: NodeId, args: &[NodeId], base: i64) -> Result<Value, EmitError> {
    let [container, index] = args_exact(id, args)?;

    if let Ok(selector) = literal_index(e, id, index) {
        return lower_get_literal(e, id, container, selector, base);
    }
    lower_get_gather(e, id, container, index, base)
}

/// The literal-selector fast path — see [`lower_get`].
fn lower_get_literal(
    e: &mut Emitter,
    id: NodeId,
    container: NodeId,
    selector: i64,
    base: i64,
) -> Result<Value, EmitError> {
    let idx = selector - base;
    if idx < 0 {
        return Err(EmitError::at(id, "get/get0: index out of range"));
    }
    let idx = idx as u64;

    let v = e.lower_node(container)?;
    let len = match &v.ty {
        MlirTy::Ranked(dims) if dims.len() == 1 => dims[0],
        other => {
            return Err(EmitError::at(
                id,
                format!(
                    "get/get0: only single-selector indexing into a rank-1 tensor is supported, got {other:?}"
                ),
            ));
        }
    };
    if let Some(len) = len {
        if idx >= len {
            return Err(EmitError::at(id, "get/get0: index out of range"));
        }
    }

    let sliced = e.slice(&v, &[idx], &[idx + 1], &[1]);
    Ok(e.reshape(&sliced, MlirTy::Scalar))
}

/// The runtime-index fallback — see [`lower_get`]. Reached once
/// `literal_index` fails on `index`; supported ONLY for a rank-1 `container`
/// indexed by a runtime rank-1 `Int` tensor. Every other shape (multi-
/// selector, record/table/tuple, rank-2+ operand, a non-`Int` index) is
/// refused here rather than mislowered.
fn lower_get_gather(
    e: &mut Emitter,
    id: NodeId,
    container: NodeId,
    index: NodeId,
    base: i64,
) -> Result<Value, EmitError> {
    let operand = e.lower_node(container)?;
    let idx = e.lower_node(index)?;

    let is_rank1 = |ty: &MlirTy| matches!(ty, MlirTy::Ranked(dims) if dims.len() == 1);
    if !is_rank1(&operand.ty) || !is_rank1(&idx.ty) || idx.elem != ElemKind::Int {
        return Err(EmitError::at(
            id,
            format!(
                "get/get0: selector must be a literal integer, or (for a runtime index) \
                 a rank-1 Int tensor indexing a rank-1 tensor container; got container \
                 {:?} index {:?} ({:?})",
                operand.ty, idx.ty, idx.elem
            ),
        ));
    }
    Ok(e.gather(&operand, &idx, base))
}

/// `get`/`get0`'s selector must be a literal integer (matching how the
/// determiniser always builds it, `Node::Lit(Scalar::Int(_))`) — refused
/// otherwise rather than attempting to lower a general expression to a
/// compile-time slice bound.
fn literal_index(e: &Emitter, id: NodeId, index: NodeId) -> Result<i64, EmitError> {
    match e.node(index) {
        Node::Lit(Scalar::Int(i)) => Ok(*i),
        _ => Err(EmitError::at(
            id,
            "get/get0: selector must be a literal integer",
        )),
    }
}

// ---- in (interval membership) ------------------------------------------------

/// The §03 element sets `in(v, S)` lowers a membership test for. Every other
/// set expression refuses.
#[derive(Clone, Copy)]
enum ElemSet {
    /// §03 `interval(lo, hi)` — "denotes the closed interval $[lo, hi]$".
    Interval(NodeId, NodeId),
    /// §03 `posreals` — "$(0, +\infty]$, the positive reals including
    /// $+\infty$". OPEN at zero.
    PosReals,
    /// §03 `nonnegreals` — "$[0, +\infty]$, the non-negative reals including
    /// $+\infty$". CLOSED at zero.
    NonNegReals,
}

/// `in(v, S)` (spec §06 membership predicate `_ in R`) over a §03 element set,
/// or over `cartpow(S, n)` of one for a vector variate.
///
/// A supported `S` is [`ElemSet`]: `interval(lo, hi)`, `posreals`,
/// `nonnegreals`. Every other set expression (`reals`, `unitinterval`,
/// `integers`, a `cartprod`, a `stdsimplex`) refuses rather than lowering an
/// approximation — the determiniser's image gates emit only these.
fn lower_in(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [v_id, set_id] = args_exact(id, args)?;
    let v = e.lower_node(v_id)?;

    // §03 `cartpow(S, size)`: "the Cartesian power of `S` with shape `size`" —
    // membership holds when EVERY cell is in `S`, so the per-cell predicate is
    // all-reduced to the scalar `i1` the enclosing `ifelse` needs.
    if let Some((elem_id, n_id)) = cartpow_parts(e, set_id) {
        let dims = match &v.ty {
            MlirTy::Ranked(dims) if dims.len() == 1 => dims.clone(),
            other => {
                return Err(EmitError::at(
                    id,
                    format!(
                        "'in' over cartpow: only a rank-1 (vector) point is supported, got {other:?}"
                    ),
                ));
            }
        };
        // A literal power that disagrees with the point's own static length is a
        // type error upstream, not something to lower against one of the two.
        if let (Node::Lit(Scalar::Int(n)), Some(len)) = (e.node(n_id), dims[0]) {
            if *n < 0 || (*n as u64) != len {
                return Err(EmitError::at(
                    id,
                    format!("'in' over cartpow({n}): point has length {len}"),
                ));
            }
        }
        let set = classify_elem_set(e, id, elem_id)?;
        let per_cell = elem_membership(e, id, &v, set)?;
        return Ok(e.reduce_all(&per_cell));
    }

    let set = classify_elem_set(e, id, set_id)?;
    elem_membership(e, id, &v, set)
}

/// The membership predicate for one [`ElemSet`], elementwise at `v`'s shape.
///
/// `interval(lo, hi)` lowers to two separate comparisons, `v ≥ lo` and
/// `v ≤ hi`, ANDed together. NOT the algebraic identity `(v - lo) · (hi - v)
/// ≥ 0`: that product is `0 · inf = NaN` at `v = lo` when `hi` is infinite,
/// so `interval(lo, inf)` mis-lowered FALSE at its own closed lower bound —
/// `truncate(Normal, interval(0, inf))` scored `-inf` at `y = 0` where the
/// finite half-normal density is owed (measured, Enzyme-JAX f32; see
/// `flatppl-dev/TODO-flatppl-rust.md`'s `ge`/`le` follow-up).
///
/// `posreals`/`nonnegreals` lower to one comparison against zero, and the
/// DIRECTION is not interchangeable: §03 makes `posreals` $(0, +\infty]$ (open
/// at zero, so `GT`) and `nonnegreals` $[0, +\infty]$ (closed, so `GE`). Both
/// admit `+inf` (§03 "Note on infinities"), which both directions already
/// accept; `NaN` compares false in either, i.e. is outside the set.
fn elem_membership(
    e: &mut Emitter,
    id: NodeId,
    v: &Value,
    set: ElemSet,
) -> Result<Value, EmitError> {
    match set {
        ElemSet::PosReals => {
            let zero = e.constant(0.0, v.ty.clone());
            Ok(e.compare("GT", v, &zero))
        }
        ElemSet::NonNegReals => {
            let zero = e.constant(0.0, v.ty.clone());
            Ok(e.compare("GE", v, &zero))
        }
        ElemSet::Interval(lo_id, hi_id) => {
            let lo = e.lower_node(lo_id)?;
            let hi = e.lower_node(hi_id)?;
            let lo = broadcast_to(e, id, &lo, &v.ty)?;
            let hi = broadcast_to(e, id, &hi, &v.ty)?;
            let above_lo = e.compare("GE", v, &lo);
            let below_hi = e.compare("LE", v, &hi);
            Ok(e.and(&above_lo, &below_hi))
        }
    }
}

/// Classify `S` as an [`ElemSet`], refusing any other set expression.
fn classify_elem_set(e: &Emitter, id: NodeId, set_id: NodeId) -> Result<ElemSet, EmitError> {
    let refuse = || {
        EmitError::at(
            id,
            "'in': only an interval(lo, hi), posreals or nonnegreals set is supported \
             (optionally under one cartpow)",
        )
    };
    match e.node(set_id) {
        Node::Const(sym) => match e.resolve(*sym) {
            "posreals" => Ok(ElemSet::PosReals),
            "nonnegreals" => Ok(ElemSet::NonNegReals),
            _ => Err(refuse()),
        },
        Node::Call(c) => match c.head {
            CallHead::Builtin(sym) if e.resolve(sym) == "interval" => args_exact::<2>(id, &c.args)
                .map(|[lo, hi]| ElemSet::Interval(lo, hi))
                .map_err(|_| refuse()),
            _ => Err(refuse()),
        },
        _ => Err(refuse()),
    }
}

/// Destructure `cartpow(S, n)` into `(S, n)`; `None` for anything else.
fn cartpow_parts(e: &Emitter, set_id: NodeId) -> Option<(NodeId, NodeId)> {
    match e.node(set_id) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(sym) if e.resolve(sym) == "cartpow" && c.args.len() == 2 => {
                Some((c.args[0], c.args[1]))
            }
            _ => None,
        },
        _ => None,
    }
}
