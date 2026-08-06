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
//! `mul` — so [`classify_bare_mul`]/[`lower_bare_mul`] handle a bare `mul`
//! over a rank-2 left operand as one `stablehlo.dot_general`, and the `"mul"`
//! entry in the map below stays purely elementwise. That dispatch lives in
//! `Emitter::lower_node`, which [`Emitter::lower_broadcast`] bypasses, so a
//! `.*`-derived `mul` can never reach it.
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
        "ifelse" => lower_ifelse(e, id, args),
        "inf" => lower_inf(e, id, args),
        "pi" => lower_pi(e, id, args),
        "logsumexp" => lower_logsumexp(e, id, args),
        "vector" => lower_vector(e, id, args),
        "sum" => unary(e, id, args, Emitter::reduce_sum),
        // §07 reductions `maximum`/`minimum` ($\max_i x_i$ / $\min_i x_i$ over
        // a real array) — NOT §07's binary `max`/`min`, which this map does not
        // lower.
        "maximum" => lower_extremum(e, id, args, Extremum::Max),
        "minimum" => lower_extremum(e, id, args, Extremum::Min),
        "fill" => lower_fill(e, id, args),
        "get0" => lower_get(e, id, args, 0),
        "get" => lower_get(e, id, args, 1),
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
        "land" => lower_land(e, id, args),
        "iszero" => lower_iszero(e, id, args),
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

fn binary<'m>(
    e: &mut Emitter<'m>,
    id: NodeId,
    args: &[NodeId],
    op: fn(&mut Emitter<'m>, &Value, &Value) -> Value,
) -> Result<Value, EmitError> {
    let [a, b] = args_exact(id, args)?;
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
    /// A rank-2 lhs against a rank-2 or rank-1 rhs: one `stablehlo.dot_general`.
    MatrixProduct,
    /// At least one operand is a scalar (or has no inferred type): the ordinary
    /// elementwise multiply, which broadcasts the scalar side.
    Elementwise,
    /// Both operands are non-scalar but the pair is not a product this emitter
    /// implements — and §07 gives `*` NO elementwise meaning, so lowering it
    /// elementwise would answer a different question than the model asked.
    Undefined,
}

/// Classify a bare `mul`'s operands by their inferred FlatPDL types — the same
/// information `infer`'s `mul_type` reads — so no operand is lowered to decide.
///
/// Only `Some(2)` on the LEFT makes a product, matching `mul_type`'s own
/// matrix·matrix / matrix·vector arms. Every other non-scalar pair is
/// [`BareMul::Undefined`]: §07 gives `*` on two vectors a meaning only WITH a
/// transpose ("the product of a transposed vector and a non-transposed vector is
/// a scalar") and none at all for rank-3 operands, and `mul_type` agrees by
/// returning `Type::Deferred` for both. An operand whose type is absent
/// (a freshly synthesized determiniser node before re-inference) or scalar
/// classifies as [`BareMul::Elementwise`], which is what keeps the determiniser's
/// own `mul(literal, vector)` idiom lowering.
fn classify_bare_mul(e: &Emitter, args: &[NodeId]) -> BareMul {
    let [a, b] = match <[NodeId; 2]>::try_from(args) {
        Ok(pair) => pair,
        // Wrong arity: let `binary`'s `args_exact` produce the arity message.
        Err(_) => return BareMul::Elementwise,
    };
    match (array_rank(e, a), array_rank(e, b)) {
        (Some(2), Some(2)) | (Some(2), Some(1)) => BareMul::MatrixProduct,
        (Some(_), Some(_)) => BareMul::Undefined,
        _ => BareMul::Elementwise,
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
/// The `Undefined` refusal also covers the `TVector` products §07 defines
/// (`transpose(v) * w`): this emitter has no inner-product lowering, and
/// refusing is right where silently emitting an elementwise multiply would be a
/// wrong number. Whoever adds a `transpose` lowering must add the `TVector` arms
/// to [`classify_bare_mul`] and a `[n]x[n] -> scalar` contraction with it; until
/// then the shape refuses here rather than at `transpose`.
pub(crate) fn lower_bare_mul(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    match classify_bare_mul(e, args) {
        BareMul::MatrixProduct => lower_matrix_product(e, id, args),
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
                     \"Linear algebra\" defines `*` as the matrix product (a matrix against a \
                     matrix or a vector), and for vectors only through a transpose. Write `.*` \
                     for an elementwise product, or `transpose(a) * b` for an inner product \
                     (which this emitter does not yet lower)",
                    shape(args[0]),
                    shape(args[1])
                ),
            ))
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
    /// [`array_rank`]/`crate::types::mlir_type_of` flatten them. `None` is a
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
/// - `divide`: "scalars, vector-scalar, matrix-scalar (real or complex)"
///   (flatppl-design#75). §05 "No implicit operator broadcasting" states the same
///   constraint directly — "`/` requires a scalar divisor, and `^` is
///   scalar-only" — so the DIVISOR is the discriminator here: a rank-1 or rank-2
///   dividend over a scalar divisor is scalar multiplication by the reciprocal,
///   sound, and lowers as the ordinary scalar-broadcast divide. `scalar / vector`
///   is NOT in the domain; an elementwise reciprocal is `./`'s job.
///
///   A rank-3 dividend refuses, which is the one place the two clauses can be read
///   differently: §05 constrains only the divisor (satisfied), while §07 enumerates
///   `vector-scalar` and `matrix-scalar` and stops. §07's table is the precise
///   statement and §05 the prose summary — for `mul`, §05's "supports matrix and
///   matrix–vector multiplication" is likewise less complete than §07's row, which
///   [`classify_bare_mul`] follows strictly. So this guard follows §07 and refuses
///   rank 3. Dividing a rank-3 array by a scalar IS sound maths, so admitting it
///   would be a spec-row change, not a guard relaxation.
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
    const DIVIDE_DOMAIN: &str = "\"scalars, vector-scalar, matrix-scalar\"";
    match head {
        "pow" if sa.is_array() || sb.is_array() => refuse("\"scalars\""),
        // A known-array DIVISOR is proof enough on its own, whatever the dividend
        // is: it rules out `scalar / vector` and `vector / vector` together.
        "divide" if sb.is_array() => refuse(DIVIDE_DOMAIN),
        // Dividend rank 1 or 2 over a scalar divisor is in the amended row; a
        // higher-rank dividend is not enumerated, so it refuses.
        "divide" if sa.is_array() && !matches!(&sa, ArithShape::Array(d) if d.len() <= 2) => {
            refuse(DIVIDE_DOMAIN)
        }
        "add" | "sub" if sa.differs_from(&sb) => refuse("\"scalars or arrays of same shape\""),
        _ => binary(e, id, args, op),
    }
}

/// The rank of `id`'s inferred `Array`/`TVector` type, flattening a nested
/// element chain the way `crate::types::mlir_type_of` does — `None` for a scalar,
/// an untyped node, or any other type.
fn array_rank(e: &Emitter, id: NodeId) -> Option<usize> {
    fn rank(ty: &Type) -> Option<usize> {
        match ty {
            Type::Array { shape, elem } => Some(shape.len() + rank(elem).unwrap_or(0)),
            Type::TVector { elem, .. } => Some(1 + rank(elem).unwrap_or(0)),
            _ => None,
        }
    }
    rank(e.type_of(id)?)
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

// ---- ifelse / inf -----------------------------------------------------------

fn lower_ifelse(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [c, a, b] = args_exact(id, args)?;
    require_predicate_head(e, c, "ifelse condition")?;
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
const PREDICATE_HEADS: &[&str] = &["in", "compare", "lt", "gt", "le", "ge", "land", "iszero"];

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
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    // `Emitter::compare` reconciles its operands through `broadcast_pair` exactly
    // as `Emitter::binary` does, and is equally infallible.
    require_broadcastable(id, &a, &b)?;
    Ok(e.compare(dir, &a, &b))
}

/// §07 `land` (`a && b`, over `booleans`) — `stablehlo.and` over two `i1`
/// predicates. Both operands must be [`PREDICATE_HEADS`] calls, and must share
/// a shape: [`Emitter::and`] renders ONE type for both operands and the result,
/// so a mismatched pair would emit ill-typed text.
fn lower_land(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a_id, b_id] = args_exact(id, args)?;
    require_predicate_head(e, a_id, "land operand")?;
    require_predicate_head(e, b_id, "land operand")?;
    let a = e.lower_node(a_id)?;
    let b = e.lower_node(b_id)?;
    if a.ty != b.ty {
        return Err(EmitError::at(
            id,
            format!(
                "land: operands must have the same shape, got {:?} and {:?}",
                a.ty, b.ty
            ),
        ));
    }
    Ok(e.and(&a, &b))
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
