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

use flatppl_core::{CallHead, Node, NodeId, Scalar};

use crate::emitter::Emitter;
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
        // §07 comparison functions `lt`/`gt` ($a < b$ / $a > b$ over `reals`).
        "lt" => lower_compare(e, id, args, "LT"),
        "gt" => lower_compare(e, id, args, "GT"),
        "land" => lower_land(e, id, args),
        "iszero" => lower_iszero(e, id, args),
        // `record(...)` is not a tensor — handled structurally by the mode
        // builder (a record-typed model input's fields become separate
        // tensor args), never reached here in a well-formed lowering.
        "record" => Err(EmitError::at(id, "record has no tensor form")),
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
    Ok(op(e, &a, &b))
}

// ---- ifelse / inf -----------------------------------------------------------

fn lower_ifelse(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [c, a, b] = args_exact(id, args)?;
    require_predicate_head(e, c, "ifelse condition")?;
    let c = e.lower_node(c)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    Ok(e.select(&c, &a, &b))
}

/// The predicate-producing builtin heads this map lowers to an `i1` value.
/// [`Emitter::select`] and [`Emitter::and`] unconditionally render their
/// predicate operands as `i1`, so handing either any other node (e.g. a bare
/// `Lit(Bool)`, which lowers as a plain `tensor<f32>` `dense<1.0>` via
/// `constant`) would make the declared `i1` operand disagree with the actual
/// emitted type, producing ill-typed StableHLO.
const PREDICATE_HEADS: &[&str] = &["in", "compare", "lt", "gt", "land", "iszero"];

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

/// §07 `real` — "returns `x` for real `x`". Value identity: NO rounding,
/// clamping or truncation. The one thing it may emit is the §03
/// `booleans ⊂ integers ⊂ reals` embedding when the operand really is
/// integer-typed, which is exact; [`Emitter::convert`] emits nothing at all
/// for an already-real operand. The decision is made on the operand's own
/// lowered kind, not on the node's name — the determiniser wraps `round` in
/// `real` precisely to stop `exp(round(x))` from typing as `integers` and
/// being evaluated in integer arithmetic.
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
    let x = e.convert(&x, kind);
    Ok(e.broadcast_in_dim(&x, &[], ty))
}

fn lower_inf(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    args_exact::<0>(id, args)?;
    Ok(e.inf(MlirTy::Scalar))
}

/// §03 `pi` — "the mathematical constant $\pi$". Gate vocabulary: `atan`'s
/// image endpoint is `pi / 2` (`crate`-external, `determinizer::invert`).
/// `f64`'s `Display` is shortest-round-trip, so the emitted decimal literal is
/// exact at both dtypes.
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
/// `interval(lo, hi)` lowers to a SINGLE `compare` via the closed-interval
/// algebraic identity `v ∈ [lo, hi] ⟺ (v - lo) · (hi - v) ≥ 0` (zero, i.e.
/// included, exactly at either boundary; negative outside it, for `lo ≤ hi`).
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
            let below = e.sub(v, &lo);
            let above = e.sub(&hi, v);
            let product = e.mul(&below, &above);
            let zero = e.constant(0.0, v.ty.clone());
            Ok(e.compare("GE", &product, &zero))
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
