//! Spec §07 "Boolean reductions", "Cumulative operations" and the order
//! statistics.
//!
//! A separate module from [`crate::norms`] for the reason that module's own doc
//! gives: several emitter waves append heads concurrently, and one module per wave
//! keeps the textual conflict surface at zero. The shared guards are re-used from
//! there rather than copied.
//!
//! **`lany`/`lall` reduce in `i1`, with no promotion.** This is the deliberate
//! contrast with `sum`: §03 "Bool" promotes a boolean "in arithmetic contexts",
//! and a 1-bit `stablehlo.add` is parity rather than a count, which is why
//! [`crate::norms`] widens a boolean operand before every reduction it wires.
//! §07 defines `lany` as "the `lor`-reduction of its input" and `lall` as "the
//! `land`-reduction", `booleans` is closed under both, and a truth value is not an
//! arithmetic context — so `stablehlo.or`/`stablehlo.and` over `i1` compute exactly
//! what §07 says, and widening would be the mistake.
//!
//! `median` and `quantile` sort each array cell and interpolate order statistics.

use flatppl_core::{NodeId, Type};

use crate::emitter::{AxisReduce, Emitter};
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::norms::{require_not_table_vector, table_reduction_in_tensor_position};
use crate::ops::args_exact;
use crate::refuse::EmitError;

/// §07 "Boolean reductions" — `lany` is the `lor`-reduction, `lall` the
/// `land`-reduction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoolReduce {
    Any,
    All,
}

impl BoolReduce {
    fn spec_name(self) -> &'static str {
        match self {
            BoolReduce::Any => "lany",
            BoolReduce::All => "lall",
        }
    }

    /// The `stablehlo` combine and its identity. `false` is the identity of
    /// disjunction and `true` of conjunction, which is also what each head answers
    /// over an EMPTY array — `stablehlo.reduce` over a zero-length axis returns its
    /// init, so no special case is needed and no value is fabricated. The owner's
    /// zero-size-arrays ruling of 2026-08-20
    /// (`flatppl-dev/empty-arrays-ruling.md`, sub-ruling 2) fixes exactly this —
    /// "`lany([])` = `false`, `lall([])` = `true` … stand (forced identities)" — so
    /// these are spec-backed rather than a convention awaiting one. Julia's
    /// `any(Bool[])` / `all(Bool[])` agree. Both executed.
    fn combine(self) -> (&'static str, &'static str) {
        match self {
            BoolReduce::Any => ("stablehlo.or", "false"),
            BoolReduce::All => ("stablehlo.and", "true"),
        }
    }
}

// ---- §07 boolean reductions ----------------------------------------------------

/// §07 "Boolean reductions" `lany`/`lall` over a boolean array — one `i1`
/// [`Emitter::reduce_boolean`] per axis.
///
/// The operand must be BOOLEAN. §03's `booleans ⊂ integers ⊂ reals` runs the other
/// way, so a real or integer array is not in §07's "boolean arrays" domain and
/// there is no conversion to make that is not a truthiness convention §07 never
/// states. (The js engine reads truthiness off a real column here and its own
/// review flags that as a gap; this refuses instead.)
///
/// The reachable operands are a boolean ABI input
/// (`elementof(cartpow(booleans, [n]))`, which `crate::types` maps to
/// `tensor<nxi1>`) and a DOTTED comparison mask (`v .> 3.0`, which
/// [`Emitter::compare`] lowers to `tensor<nxi1>` with [`ElemKind::Bool`]). The bare
/// `gt(v, 3.0)` is not one: `infer` refuses it, because §07 gives the comparisons a
/// scalar domain.
pub(crate) fn lower_boolean_reduction(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: BoolReduce,
) -> Result<Value, EmitError> {
    let [xs_id] = args_exact(id, args)?;
    if matches!(e.type_of(xs_id), Some(Type::Table { .. })) {
        return Err(table_reduction_in_tensor_position(id));
    }
    let xs = e.lower_node(xs_id)?;
    boolean_reduction_value(e, id, xs, which)
}

/// [`lower_boolean_reduction`] over an already-lowered operand.
pub(crate) fn boolean_reduction_value(
    e: &mut Emitter,
    id: NodeId,
    xs: Value,
    which: BoolReduce,
) -> Result<Value, EmitError> {
    let head = which.spec_name();
    if !matches!(&xs.ty, MlirTy::Ranked(dims) if dims.iter().all(Option::is_some)) {
        return Err(EmitError::at(
            id,
            format!(
                "{head}: §07 reduces an ARRAY, so the operand must be a statically-shaped array, \
                 got {:?}",
                xs.ty
            ),
        ));
    }
    if xs.elem != ElemKind::Bool {
        return Err(EmitError::at(
            id,
            format!(
                "{head}: §07 \"Boolean reductions\" gives this head the domain \"boolean arrays\", \
                 and got an array of {:?}. §03's `booleans ⊂ integers ⊂ reals` does not admit a \
                 real or integer array here — the inclusion runs the other way — and reading \
                 truthiness off one would be a convention §07 does not state. Compare it first: \
                 `{head}(v .> 0.0)`",
                xs.elem
            ),
        ));
    }
    let (op, identity) = which.combine();
    Ok(e.reduce_boolean(op, identity, &xs))
}

// ---- §07 cumulative extrema ----------------------------------------------------

/// §07 "Cumulative operations" `cummax`/`cummin` over a vector — one
/// [`Emitter::prefix_scan_extremum`], the same `stablehlo.reduce_window` pass
/// `cumsum`/`cumprod` take, with a `maximum`/`minimum` combine and the matching
/// ±inf (or integer-extreme) seed.
///
/// Unlike [`crate::norms::lower_cumulative`], this does NOT promote a boolean
/// operand: `infer`'s `SameAsArg(0)` row is exact for these two, because a running
/// maximum SELECTS an element and performs no arithmetic, so §03 "Bool"'s
/// promotion sentence does not reach it. That makes the boolean case a refusal
/// rather than a widening — see [`Emitter::prefix_scan_extremum`], which carries
/// the IREE limitation that forces it.
///
/// An EMPTY vector scans to the empty vector, handled here for
/// [`crate::norms::lower_cumulative`]'s reason: `stablehlo.reduce_window` has no
/// `window_dimensions = 0` form, and the prefix sequence of an empty sequence is
/// empty, so returning the operand answers §07's shape-preserving row without
/// emitting an op that cannot verify. The owner's zero-size-arrays ruling of
/// 2026-08-20 (`flatppl-dev/empty-arrays-ruling.md`, sub-ruling 2) makes that the
/// rule for the whole family: "`softmax`, `logsoftmax`, `cumsum`, `cumprod`,
/// `l1unit`, `l2unit` all map empty to empty", uniform "across all vector-valued
/// heads", which these two are.
pub(crate) fn lower_cumulative_extremum(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    kind: AxisReduce,
) -> Result<Value, EmitError> {
    let head = match kind {
        AxisReduce::Max => "cummax",
        AxisReduce::Min => "cummin",
        other => panic!("lower_cumulative_extremum is only for Max/Min, got {other:?}"),
    };
    let [xs_id] = args_exact(id, args)?;
    require_not_table_vector(e, id, head, xs_id)?;
    let xs = e.lower_node(xs_id)?;
    let n = require_static_vector(id, head, &xs)?;
    if n == 0 {
        return Ok(xs);
    }
    e.prefix_scan_extremum(id, kind, &xs)
}

// ---- §07 the infinity norm -----------------------------------------------------

/// §07 "Norms and normalization" `linfnorm`: $\max_i \lvert v_i\rvert$ — one
/// `stablehlo.abs` then a `maximum` reduce, the shape
/// [`crate::norms::lower_norm`]'s $\ell^1$ arm takes with the combine swapped.
///
/// The operand is widened to `Real` first, exactly as its two siblings are and for
/// the same reason: §07's domain is "real/complex vectors", §03 admits an integer
/// or boolean vector inside it, and `infer` types the call `Scalar(Real)`, so the
/// emitted return type must be the float dtype whatever arrived.
///
/// An EMPTY vector gives `0.0`, and this one IS a special case rather than a
/// fall-through. The reduce's own answer over a zero-length axis is its identity,
/// `-inf` (executed) — a NEGATIVE value for a quantity §07 declares a norm, and one
/// that appears in neither the input nor the spec. `0.0` is what the siblings give
/// (`l1norm([])`/`l2norm([])`, the empty sum), what both oracles give
/// (`np.linalg.norm([], inf)` and Julia `norm(Float64[], Inf)` are each `0.0`), and
/// what the owner's zero-size-arrays ruling of 2026-08-20 fixes —
/// "`l1norm`/`l2norm`/`linfnorm` = `0` stand (forced identities)"
/// (`flatppl-dev/empty-arrays-ruling.md`, sub-ruling 2).
pub(crate) fn lower_linf_norm(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [v_id] = args_exact(id, args)?;
    let v = e.lower_node(v_id)?;
    let n = require_static_vector(id, "linfnorm", &v)?;
    if n == 0 {
        return Ok(e.scalar(0.0));
    }
    let v = e.convert(&v, ElemKind::Real);
    let a = e.abs(&v);
    e.reduce_trailing_axes(id, AxisReduce::Max, &a, 1)
}

// ---- §07 order statistics -----------------------------------------------------

pub(crate) fn lower_order_statistic(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    median: bool,
) -> Result<Value, EmitError> {
    let (xs_id, p_id) = if median {
        let [xs] = args_exact(id, args)?;
        (xs, None)
    } else {
        let [xs, p] = args_exact(id, args)?;
        (xs, Some(p))
    };
    if matches!(e.type_of(xs_id), Some(Type::Table { .. })) {
        return Err(table_reduction_in_tensor_position(id));
    }
    let xs = e.lower_node(xs_id)?;
    let p = p_id.map(|p| e.lower_node(p)).transpose()?;
    order_statistic_value(e, id, xs, p)
}

/// Reduce every cell axis, leaving callable batch axes intact. NaNs are replaced
/// before sorting to keep the comparator ordered, then propagated per cell.
pub(crate) fn order_statistic_value(
    e: &mut Emitter,
    id: NodeId,
    xs: Value,
    p: Option<Value>,
) -> Result<Value, EmitError> {
    let dims = match e.cell_ty(&xs) {
        MlirTy::Ranked(dims) if !dims.is_empty() && dims.iter().all(Option::is_some) => dims,
        other => {
            return Err(EmitError::at(
                id,
                format!("order statistic requires a statically shaped array, got {other:?}"),
            ));
        }
    };
    let dims: Vec<u64> = dims.into_iter().map(Option::unwrap).collect();
    let n = crate::norms::checked_shape_product(id, "order statistic", &dims)?;
    if n == 0 {
        return Err(EmitError::at(
            id,
            "order statistic over an empty array is undefined (spec §07)",
        ));
    }
    // The index calculation uses the emitted floating dtype and integer width.
    let exact_limit = match e.dtype() {
        crate::Dtype::F32 => 1_u64 << 24,
        crate::Dtype::F64 => 1_u64 << 53,
    };
    if n > exact_limit {
        return Err(EmitError::at(
            id,
            "order statistic array is too large for exact emitted indices",
        ));
    }
    let xs = e.convert(&xs, ElemKind::Real);
    let xs = e.reshape_cell(&xs, MlirTy::Ranked(vec![Some(n)]));
    let valid = e.compare("EQ", &xs, &xs);
    let all_valid = e.reduce_boolean("stablehlo.and", "true", &valid);
    let zero = e.constant_like(0.0, &xs);
    let cleaned = e.select(&valid, &xs, &zero);
    let sorted = e.sort_vector(&cleaned);
    let p = p.unwrap_or_else(|| e.scalar(0.5));
    if e.cell_ty(&p) != MlirTy::Scalar {
        return Err(EmitError::at(id, "quantile: p must be a scalar"));
    }
    let p = e.convert(&p, ElemKind::Real);
    let p_valid = e.compare("EQ", &p, &p);
    let zero = e.scalar(0.0);
    let p_safe = e.select(&p_valid, &p, &zero);
    let count = e.scalar((n - 1) as f64);
    let position = e.mul(&p_safe, &count);
    let floor = e.floor(&position);
    let fraction = e.sub(&position, &floor);
    let low_index = e.convert(&floor, ElemKind::Int);
    let one = e.int_value_const(1);
    let upper_index = e.add(&low_index, &one);
    let last = e.int_value_const((n - 1) as i64);
    let past = e.compare("GT", &upper_index, &last);
    let upper_index = e.select(&past, &last, &upper_index);
    let low = e.gather_cells(&sorted, &low_index, 0);
    let high = e.gather_cells(&sorted, &upper_index, 0);
    let one = e.scalar(1.0);
    let complement = e.sub(&one, &fraction);
    let left = e.mul(&complement, &low);
    let right = e.mul(&fraction, &high);
    let interpolated = e.add(&left, &right);
    let zero = e.scalar(0.0);
    let exact = e.compare("EQ", &fraction, &zero);
    let result = e.select(&exact, &low, &interpolated);
    let all_valid = e.and(&all_valid, &p_valid);
    let nan = e.scalar(f64::NAN);
    Ok(e.select(&all_valid, &result, &nan))
}

// ---- shared guard --------------------------------------------------------------

/// [`crate::norms::require_static_vector`]'s twin, for the same §07 "vectors"
/// domain. Duplicated rather than shared to keep this wave's module free of edits
/// to `crate::norms`, whose sibling branch is concurrently changing.
fn require_static_vector(id: NodeId, head: &str, v: &Value) -> Result<u64, EmitError> {
    match &v.ty {
        MlirTy::Ranked(dims) if dims.len() == 1 => {
            if let Some(n) = dims[0] {
                return Ok(n);
            }
        }
        _ => {}
    }
    Err(EmitError::at(
        id,
        format!(
            "{head}: §07 gives this head the domain \"vectors\", so its operand must be a \
             statically-sized rank-1 array, got {:?}",
            v.ty
        ),
    ))
}
