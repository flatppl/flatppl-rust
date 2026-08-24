//! Spec §07 "Boolean reductions", "Cumulative operations" and the order
//! statistics — the heads the `missing-reductions` spec draft (flatppl-design
//! `ee4c6fb`) adds to §07, four of which lower here and two of which refuse.
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
//! **`median`/`quantile` refuse.** Both are order statistics, and this crate has no
//! sort: `stablehlo.sort` appears nowhere in it, and neither does a top-k. See
//! [`refuse_order_statistic`] for what a lowering would cost and why answering
//! without one would be worse than refusing.

use flatppl_core::{NodeId, Type};

use crate::emitter::{AxisReduce, Emitter};
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::norms::table_reduction_refusal;
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

/// Which order statistic a refused head computes, for [`refuse_order_statistic`].
#[derive(Clone, Copy)]
pub(crate) enum OrderStatistic {
    Median,
    Quantile,
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
    let head = which.spec_name();
    let [xs_id] = args_exact(id, args)?;
    if matches!(e.type_of(xs_id), Some(Type::Table { .. })) {
        return Err(table_reduction_refusal(id, head));
    }
    let xs = e.lower_node(xs_id)?;
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
    if matches!(e.type_of(xs_id), Some(Type::Table { .. })) {
        return Err(table_reduction_refusal(id, head));
    }
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

// ---- §07 order statistics: refused ---------------------------------------------

/// §07 `median(xs)` and `quantile(xs, p)` have no lowering in this emitter.
///
/// Both are ORDER statistics: §07 defines `median` on "the order statistics
/// $x_{(1)} \le \dots \le x_{(n)}$" and `quantile` as "linear interpolation between
/// the order statistics of `xs`". Every route to an order statistic needs the input
/// ranked, and this crate has no ranking primitive — it emits no `stablehlo.sort`
/// anywhere, and no top-k. (The name occurs only in prose like this and inside the
/// refusal message below; no builder produces the op.)
///
/// The two ways to add one, and why neither is taken here rather than deliberately:
///
/// - **A `stablehlo.sort` builder.** A new region-carrying op (the shape
///   [`Emitter::prefix_scan`] has for `reduce_window`) plus a comparator region,
///   plus a decision about where NaN sorts, plus a `p`-dependent index that is a
///   RUNTIME value in general — so a `dynamic_slice`, not a static one. That is a
///   feature, not a wiring.
/// - **Rank-select without a sort**: $\mathrm{rank}(i) = \#\{j : x_j < x_i\} +
///   \#\{j < i : x_j = x_i\}$ over an $n \times n$ comparison matrix, then select
///   the wanted rank. Exact over the reals, and O(n²) in emitted tensor ops rather
///   than in unrolled scalars. Unguarded it is silently WRONG in the presence of NaN:
///   every comparison against NaN is false, so two or more elements collide at the
///   same rank and the "sorted" vector then contains a fabricated element. A wrong
///   number with no diagnostic is the one outcome worse than refusing. That guard is
///   UNBUILT and out of scope here, NOT unbuildable: `lany(isnan.(x))` selecting
///   between the rank-select result and NaN is a handful of ops on top of the O(n²)
///   emit, and both halves already lower (`isnan` is an elementwise cell, `lany`
///   reduces to a scalar `i1`). NaN-propagation is also what `np.median` does. So
///   this route stays open for a future wave.
///
/// So this refuses, localized to the call, and names the head. The js engine
/// implements both (its `_sortedCopy` sorts a host array), so the language has them
/// — this backend does not, and saying so is honest. Recorded in
/// `flatppl-dev/TODO-flatppl-rust.md` and
/// `flatppl-dev/stablehlo-feature-matrix.md`.
pub(crate) fn refuse_order_statistic(id: NodeId, which: OrderStatistic) -> EmitError {
    let (head, defn) = match which {
        OrderStatistic::Median => ("median", "the middle order statistic of `xs`"),
        OrderStatistic::Quantile => (
            "quantile",
            "linear interpolation between two adjacent order statistics of `xs`",
        ),
    };
    EmitError::at(
        id,
        format!(
            "{head} has no lowering here: §07 defines it as {defn}, which needs the operand \
             RANKED, and this emitter has no sort — `stablehlo.sort` is not in its vocabulary and \
             neither is a top-k. Adding one is a feature (a region-carrying op, a NaN ordering \
             rule, and a runtime index for `p`), and the sort-free rank-select alternative \
             fabricates an element on a NaN input unless it is guarded, which it is not here, so \
             this refuses rather than answering. The js engine implements both heads"
        ),
    )
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
