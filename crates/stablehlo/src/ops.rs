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
        "ifelse" => lower_ifelse(e, id, args),
        "inf" => lower_inf(e, id, args),
        "logsumexp" => lower_logsumexp(e, id, args),
        "vector" => lower_vector(e, id, args),
        "sum" => unary(e, id, args, Emitter::reduce_sum),
        "get0" => lower_get(e, id, args, 0),
        "get" => lower_get(e, id, args, 1),
        "in" => lower_in(e, id, args),
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
    require_predicate_head(e, c)?;
    let c = e.lower_node(c)?;
    let a = e.lower_node(a)?;
    let b = e.lower_node(b)?;
    Ok(e.select(&c, &a, &b))
}

/// `ifelse`'s condition must be a predicate-producing builtin call (`in`, or
/// a future `compare`) — [`Emitter::select`] unconditionally renders its
/// predicate operand as `i1`, so handing it any other node (e.g. a bare
/// `Lit(Bool)`, which lowers as a plain `tensor<f32>` `dense<1.0>` via
/// `constant`) would make `select`'s declared `i1` operand disagree with
/// `c`'s actual emitted type, producing ill-typed StableHLO. Same
/// narrow-and-refuse discipline as `get`/`get0`'s literal-selector check:
/// checked structurally against the *unlowered* condition node, before
/// `lower_node` ever runs on it.
fn require_predicate_head(e: &Emitter, cond: NodeId) -> Result<(), EmitError> {
    let is_predicate = matches!(
        e.node(cond),
        Node::Call(c) if matches!(
            c.head,
            CallHead::Builtin(sym) if matches!(e.resolve(sym), "in" | "compare")
        )
    );
    if is_predicate {
        Ok(())
    } else {
        Err(EmitError::at(
            cond,
            "ifelse condition must be a boolean predicate (in/compare)",
        ))
    }
}

fn lower_inf(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    args_exact::<0>(id, args)?;
    Ok(e.inf(MlirTy::Scalar))
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

/// Broadcast a `Scalar` value `a` up to `ty`'s shape via
/// [`Emitter::broadcast_in_dim`] when the shapes differ; returns `a`
/// unchanged (no op emitted) when they already match — e.g. `logsumexp`
/// over a length-1 vector, or `in`'s bound already matching a scalar
/// variate. Refuses rather than mis-emitting a shape-mismatched op if `a`
/// isn't a `Scalar` to begin with: broadcasting a *ranked* operand up to a
/// bigger shape needs an explicit dimension mapping this emitter has no
/// caller for yet.
fn broadcast_to(e: &mut Emitter, id: NodeId, a: &Value, ty: &MlirTy) -> Result<Value, EmitError> {
    if &a.ty == ty {
        Ok(a.clone())
    } else if a.ty == MlirTy::Scalar {
        Ok(e.broadcast_in_dim(a, &[], ty.clone()))
    } else {
        Err(EmitError::at(
            id,
            format!("shape mismatch: cannot broadcast {:?} to {ty:?}", a.ty),
        ))
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

/// `in(v, S)` (spec §06 membership predicate `_ in R`): only `S =
/// interval(lo, hi)` is supported — refuses any other set expression (e.g.
/// the bare constants `reals`/`posreals`, or a `cartprod`). Lowers to a
/// single `compare`, not an explicit AND of two bound checks: `MlirTy` has
/// no boolean variant to combine (see `emitter.rs`'s module doc comment), so
/// ANDing two comparisons would need a boolean-AND op this emitter doesn't
/// have. Instead uses the closed-interval algebraic identity `v ∈ [lo, hi]
/// ⟺ (v - lo) · (hi - v) ≥ 0` (zero, i.e. included, exactly at either
/// boundary; negative outside it, for `lo ≤ hi`) to reduce membership to one
/// comparison.
fn lower_in(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [v_id, set_id] = args_exact(id, args)?;
    let (lo_id, hi_id) = interval_bounds(e, id, set_id)?;

    let v = e.lower_node(v_id)?;
    let lo = e.lower_node(lo_id)?;
    let hi = e.lower_node(hi_id)?;
    let lo = broadcast_to(e, id, &lo, &v.ty)?;
    let hi = broadcast_to(e, id, &hi, &v.ty)?;

    let below = e.sub(&v, &lo);
    let above = e.sub(&hi, &v);
    let product = e.mul(&below, &above);
    let zero = e.constant(0.0, v.ty.clone());
    Ok(e.compare("GE", &product, &zero))
}

/// Destructure `S = interval(lo, hi)`, refusing any other set expression.
fn interval_bounds(e: &Emitter, id: NodeId, set_id: NodeId) -> Result<(NodeId, NodeId), EmitError> {
    let refuse = || EmitError::at(id, "'in': only an interval(lo, hi) set is supported");
    match e.node(set_id) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(sym) if e.resolve(sym) == "interval" => args_exact::<2>(id, &c.args)
                .map(|[lo, hi]| (lo, hi))
                .map_err(|_| refuse()),
            _ => Err(refuse()),
        },
        _ => Err(refuse()),
    }
}
