//! Spec §07 "Reductions" and "Norms and normalization" — the BARE heads whose
//! machinery already existed in the crate but which `crate::ops`'s builtin map
//! had no entry for.
//!
//! Every reduction here contracts through [`Emitter::reduce_trailing_axes`], the
//! same helper `crate::aggregate` uses for §04's `f_reduction`s, asked for the
//! operand's whole rank instead of an aggregation frame's trailing axes. So the
//! per-kind reduce identities come from ONE audited table rather than a second
//! copy — `prod` in particular has a multiplicative identity only there
//! ([`Emitter::reduce_axis`]'s selection is additive-only).
//!
//! `mean`/`var`/`std` are DERIVED from the sum and the statically known element
//! count exactly as §07's table defines them, matching
//! `crate::aggregate::reduce` term for term — including the $n-1$ denominator,
//! which §04's own column-wise example pins (it prints `[32, 2, 8]` for a 2-row
//! matrix; the population variance would print `[16, 1, 4]`).
//!
//! **Element kinds.** §07 gives the reductions the domain "real/complex arrays"
//! and the norms "real/complex vectors", and §03's `booleans ⊂ integers ⊂ reals`
//! puts an integer or boolean operand inside both. Each head widens its operand
//! to the kind `infer` types the CALL as, so the emitted ABI type and the
//! inferred type agree: `prod` keeps the element kind (a boolean array promotes
//! to `Int`, per §03 "Bool" and `infer::ops::reduced_scalar`), and every moment
//! and every norm is `Real`. A widening [`Emitter::convert`] is exact, so no
//! head here reaches a helper that would hardcode `Real` over an operand it
//! cannot represent.
//!
//! Separate module rather than more of `crate::ops` because a sibling branch is
//! concurrently editing that file — the map gains twelve one-line arms and the
//! bodies live here.

use flatppl_core::{NodeId, Type};

use crate::emitter::{AxisReduce, Emitter};
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::ops::args_exact;
use crate::refuse::EmitError;

/// §07 "Reductions", the whole-array heads this module wires. `sum`,
/// `maximum` and `minimum` are `crate::ops`' own (already wired); the
/// cumulative pair is [`Cumulative`], since a scan is not a reduction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BareReduction {
    Prod,
    Mean,
    Var,
    Std,
}

impl BareReduction {
    fn spec_name(self) -> &'static str {
        match self {
            BareReduction::Prod => "prod",
            BareReduction::Mean => "mean",
            BareReduction::Var => "var",
            BareReduction::Std => "std",
        }
    }

    /// Whether the head divides by an element count, so it is real arithmetic
    /// whatever the operand's kind — §07 types `mean` of an integer array real
    /// (the mean of `[1, 2]` is `1.5`), and gives `var`/`std` the domain "real
    /// arrays" outright.
    fn is_moment(self) -> bool {
        self != BareReduction::Prod
    }
}

/// §07 "Reductions" `cumsum`/`cumprod` — shape-preserving SCANS, not
/// reductions: they return a vector of the operand's own length rather than a
/// scalar, so they share no machinery with [`BareReduction`] and only sit in
/// §07's "Reductions" table by filing.
///
/// **§07 pins no scanned axis**, which is why a multi-axis operand refuses
/// instead of picking one. The row's domain is "vectors" and its description is
/// the flat sequence $(x_1, x_1+x_2, \dots)$ — one running index, no axis
/// argument and no stated default. Nothing elsewhere in the spec adds one:
/// outside that row the pair appears only in §08's Dirichlet prose and §12's
/// Stan profile mapping table. §04 closes the other door independently — its
/// `f_reduction` must be "an order-invariant vector-to-scalar reduction", which
/// a scan is neither.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cumulative {
    Sum,
    Prod,
}

impl Cumulative {
    fn spec_name(self) -> &'static str {
        match self {
            Cumulative::Sum => "cumsum",
            Cumulative::Prod => "cumprod",
        }
    }
}

/// §07 "Norms and normalization" — the two norms and the two normalizations
/// that divide by them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Norm {
    L1,
    L2,
    L1Unit,
    L2Unit,
}

impl Norm {
    fn spec_name(self) -> &'static str {
        match self {
            Norm::L1 => "l1norm",
            Norm::L2 => "l2norm",
            Norm::L1Unit => "l1unit",
            Norm::L2Unit => "l2unit",
        }
    }

    /// Whether the head returns the normalized VECTOR rather than the scalar
    /// norm.
    fn is_unit(self) -> bool {
        matches!(self, Norm::L1Unit | Norm::L2Unit)
    }
}

/// §07 "Norms and normalization" `softmax`/`logsoftmax`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Softmax {
    Plain,
    Log,
}

// ---- shared guards ------------------------------------------------------------

/// The §07 "Table reductions" refusal, for a reduction head applied to a TABLE.
///
/// §07: "When `sum`, `mean`, `var`, `std`, `prod`, `maximum`, or `minimum` is
/// applied to a table, the reduction operates column-wise and returns a record
/// whose fields are the column names and values are the per-column reductions."
/// This emitter has no record value — every [`Value`] is a tensor, and
/// `crate::ops::lower_builtin`'s `"record"` arm refuses with "record has no
/// tensor form" — so the result is not expressible here, and refusing is the
/// honest outcome rather than lowering one column and calling it the answer.
///
/// Checked BEFORE the argument is lowered, for the reason
/// `crate::ops::lower_sum` records: otherwise the argument's own refusal fires
/// first and blames the wrong thing (a `load_data` table gives the
/// column-wise-destructuring message, an `elementof` table "unsupported builtin
/// head 'elementof'", neither of which mentions the reduction).
pub(crate) fn table_reduction_refusal(id: NodeId, head: &str) -> EmitError {
    EmitError::at(
        id,
        format!(
            "a table reduction has no tensor form: §07 \"Table reductions\" makes `{head}` over a \
             table a RECORD of per-column reductions, and this emitter represents every value as a \
             tensor. Reduce one column at a time instead — `{head}(data.x)` lowers, and a table \
             input is already one argument per column"
        ),
    )
}

fn require_not_table(e: &Emitter, id: NodeId, head: &str, arg: NodeId) -> Result<(), EmitError> {
    if matches!(e.type_of(arg), Some(Type::Table { .. })) {
        return Err(table_reduction_refusal(id, head));
    }
    Ok(())
}

/// Refuse anything that is not a statically-shaped rank-1 tensor. §07 gives the
/// cumulative pair and every norm the domain "vectors" specifically, unlike the
/// reductions' "arrays", so a matrix operand has no §07 meaning to lower rather
/// than a meaning this backend declines.
///
/// A dynamic (`?`) extent is refused with the same message: the emitted window
/// and broadcast shapes are static text.
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

/// Broadcast a reduced SCALAR back over `ty`'s shape, the splat form
/// (`dims = []`) §04 "Broadcasting" defines. StableHLO's elementwise ops require
/// identical operand shapes — there is no implicit scalar broadcast.
fn splat(e: &mut Emitter, s: &Value, ty: &MlirTy) -> Value {
    e.broadcast_in_dim(s, &[], ty.clone())
}

// ---- §07 reductions -----------------------------------------------------------

/// §07 "Reductions" `prod`/`mean`/`var`/`std` over a whole array — the full
/// reduction, matching the `sum`/`maximum`/`minimum` already in the map (§07
/// writes each as a single index-free aggregate, $\prod_i x_i$ /
/// $\bar{x} = \frac{1}{n}\sum_i x_i$, over "arrays" of any rank). For a
/// per-axis contraction §07 itself points at §04: "For multi-axis array
/// contraction using these reductions, see multi-axis aggregation."
///
/// `n` is the product of every axis length, read off the lowered operand's
/// static shape.
///
/// Refuses a SCALAR operand, like `crate::ops::lower_extremum` does and unlike
/// `crate::ops::lower_sum` (which returns it unchanged through
/// [`Emitter::reduce_full`]'s zero-iteration path). §07 lists these under
/// reductions over arrays; the `sum` asymmetry is pre-existing and not changed
/// here.
pub(crate) fn lower_reduction(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: BareReduction,
) -> Result<Value, EmitError> {
    let head = which.spec_name();
    let [xs_id] = args_exact(id, args)?;
    require_not_table(e, id, head, xs_id)?;
    let xs = e.lower_node(xs_id)?;

    let dims = match &xs.ty {
        MlirTy::Ranked(dims) if dims.iter().all(Option::is_some) => dims.clone(),
        other => {
            return Err(EmitError::at(
                id,
                format!(
                    "{head}: §07 reduces an ARRAY, so the operand must be a statically-shaped \
                     array, got {other:?}"
                ),
            ));
        }
    };
    let rank = dims.len();
    let n: u64 = dims.iter().map(|d| d.expect("checked above")).product();

    // §03 "Bool": "`false` is promoted to zero and `true` to one, permitting
    // expressions such as `true + true`, `3 * false`, and `sum(mask)` to count
    // true entries". So a boolean array reaches a §07 reduction WIDENED, never
    // reduced in `i1` (where `stablehlo.multiply` is a conjunction and
    // `stablehlo.add` a wrapping 1-bit add — parity). `prod` widens to `Int`,
    // which is what `infer::ops::reduced_scalar` types `prod(bool_array)` as;
    // a moment widens to `Real` regardless.
    let target = match which {
        BareReduction::Prod if xs.elem == ElemKind::Bool => ElemKind::Int,
        BareReduction::Prod => xs.elem,
        _ => ElemKind::Real,
    };
    let xs = e.convert(&xs, target);

    if !which.is_moment() {
        return e.reduce_trailing_axes(id, AxisReduce::Prod, &xs, rank);
    }

    if n == 0 {
        return Err(EmitError::at(
            id,
            format!(
                "{head}: over an empty array this is undefined — §07 divides by the element \
                 count, which is zero here"
            ),
        ));
    }
    let total = e.reduce_trailing_axes(id, AxisReduce::Sum, &xs, rank)?;
    let count = e.scalar(n as f64);
    let mean = e.div(&total, &count);
    if which == BareReduction::Mean {
        return Ok(mean);
    }

    // §07 defines `var` with the $n-1$ (sample) denominator, so it needs two
    // elements. §04 "Relationship to broadcasting" states the same exclusion
    // from the other direction — the broadcast equivalence holds "for every
    // eligible `f_reduction` that is the identity on a one-element input; `var`
    // and `std` are undefined over a single element". `crate::aggregate::reduce`
    // refuses the identical case; the determiniser passes `var(v)` over a
    // length-1 `v` straight through, so the refusal has to land here.
    if n < 2 {
        return Err(EmitError::at(
            id,
            format!(
                "{head}: over {n} element(s) this is undefined — §07 defines it with the $n-1$ \
                 denominator, and §04 \"Relationship to broadcasting\" states that `var` and `std` \
                 are undefined over a single element"
            ),
        ));
    }
    let mean_full = splat(e, &mean, &xs.ty);
    let dev = e.sub(&xs, &mean_full);
    let sq = e.mul(&dev, &dev);
    let ssq = e.reduce_trailing_axes(id, AxisReduce::Sum, &sq, rank)?;
    let denom = e.scalar((n - 1) as f64);
    let var = e.div(&ssq, &denom);
    if which == BareReduction::Var {
        return Ok(var);
    }
    Ok(e.sqrt(&var))
}

// ---- §07 cumulative pair ------------------------------------------------------

/// §07 "Reductions" `cumsum`/`cumprod` over a vector — one
/// [`Emitter::prefix_scan`] (`stablehlo.reduce_window`), which see for why a
/// windowed pass is the honest lowering of a scan StableHLO has no op for.
///
/// The result keeps the operand's shape and, after §03's promotion, its element
/// kind — `infer`'s catalogue row for both heads is `SameAsArg(0)`, and its
/// boolean carve-out types `cumsum([true, true, false])` (`[1, 2, 2]`) with an
/// INTEGER element, which is what the promotion below produces.
pub(crate) fn lower_cumulative(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Cumulative,
) -> Result<Value, EmitError> {
    let head = which.spec_name();
    let [xs_id] = args_exact(id, args)?;
    require_not_table(e, id, head, xs_id)?;
    let xs = e.lower_node(xs_id)?;
    let n = require_static_vector(id, head, &xs)?;

    // §03's promotion, for [`lower_reduction`]'s reason: no `i1` combine
    // computes a cumulative sum or product.
    let xs = if xs.elem == ElemKind::Bool {
        e.convert(&xs, ElemKind::Int)
    } else {
        xs
    };

    // An EMPTY vector scans to the empty vector — the operand itself, no op
    // emitted. §07 gives the pair the domain "vectors" and a length-0 array is
    // one, so the result is owed rather than refusable, and it is well defined:
    // the prefix sequence of an empty sequence is empty (`np.cumsum([])` is
    // `[]`). Unlike `mean`/`var`/`std`, which this module DOES refuse over an
    // empty array, no division by the element count is involved, so there is no
    // $0/0$ to decline.
    //
    // Handled here rather than in `Emitter::prefix_scan` because
    // `stablehlo.reduce_window` cannot express it at all: the window would have
    // to be `window_dimensions = 0`, which IREE rejects outright ("expects
    // window to have positive value for 0-th window dimension"). Returning the
    // operand sidesteps the op instead of emitting one that cannot verify.
    if n == 0 {
        return Ok(xs);
    }
    let (op, identity) = match (which, xs.elem) {
        (Cumulative::Sum, ElemKind::Real) => ("stablehlo.add", "0.000000e+00"),
        (Cumulative::Sum, ElemKind::Int) => ("stablehlo.add", "0"),
        (Cumulative::Prod, ElemKind::Real) => ("stablehlo.multiply", "1.000000e+00"),
        (Cumulative::Prod, ElemKind::Int) => ("stablehlo.multiply", "1"),
        // Unreachable: `Bool` was promoted to `Int` just above.
        (_, ElemKind::Bool) => {
            return Err(EmitError::at(
                id,
                format!("{head}: a boolean operand must be promoted before the scan"),
            ));
        }
    };
    Ok(e.prefix_scan(op, identity, &xs))
}

// ---- §07 norms and normalization ----------------------------------------------

/// §07 "Norms and normalization" `l1norm`/`l2norm`/`l1unit`/`l2unit`:
/// $\sum_i \lvert v_i\rvert$, $\sqrt{\sum_i \lvert v_i\rvert^2}$, and each
/// vector divided by its own norm.
///
/// The $\ell^2$ sum is `v * v`, with NO `stablehlo.abs` — over the reals this
/// crate emits, squaring already discards the sign, so an `abs` first would be a
/// wasted op (the same reasoning `crate::ops`' `abs2` head records). $\lvert
/// v_i\rvert^2$ and $v_i^2$ differ only over the complexes, which
/// `crate::types` has no element type for.
///
/// A zero vector is NOT special-cased: `l1unit`/`l2unit` divide by zero and
/// emit the IEEE result. §07 states the quotient $v / \lVert v\rVert$ and puts
/// no exclusion on it, so gating the zero would answer a different question than
/// the one §07 asks.
pub(crate) fn lower_norm(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Norm,
) -> Result<Value, EmitError> {
    let head = which.spec_name();
    let [v_id] = args_exact(id, args)?;
    let v = e.lower_node(v_id)?;
    require_static_vector(id, head, &v)?;
    // §07's domain is "real/complex vectors" and §03 admits an integer or
    // boolean vector inside it; `infer` types `l1norm`/`l2norm` `Scalar(Real)`
    // and the unit pair a real array, so widen once here and let every op below
    // run in the kind the call is typed as.
    let v = e.convert(&v, ElemKind::Real);

    let norm = match which {
        Norm::L1 | Norm::L1Unit => {
            let a = e.abs(&v);
            e.reduce_trailing_axes(id, AxisReduce::Sum, &a, 1)?
        }
        Norm::L2 | Norm::L2Unit => {
            let sq = e.mul(&v, &v);
            let ssq = e.reduce_trailing_axes(id, AxisReduce::Sum, &sq, 1)?;
            e.sqrt(&ssq)
        }
    };
    if !which.is_unit() {
        return Ok(norm);
    }
    let d = splat(e, &norm, &v.ty);
    Ok(e.div(&v, &d))
}

/// §07 "Norms and normalization" `softmax`/`logsoftmax`:
/// $(e^{v_i} / \sum_j e^{v_j})_i$ and $(v_i - \log \sum_j e^{v_j})_i$.
///
/// Both are MAX-SHIFTED. With $m = \max_j v_j$ and $s = \sum_j e^{v_j - m}$,
/// $\sum_j e^{v_j} = e^m s$, so
/// $e^{v_i}/\sum_j e^{v_j} = e^{v_i - m}/s$ and
/// $v_i - \log\sum_j e^{v_j} = (v_i - m) - \log s$. Both identities are exact
/// in the reals and every exponent is $\le 0$, so nothing overflows for a
/// large-magnitude `v` the way the naive $e^{v_i}/\sum e^{v_j}$ would.
///
/// `softmax` DIVIDES by `s` rather than going through `exp(v - logsumexp(v))`.
/// The two are mathematically equal, but the `logsumexp` route takes a `log`
/// and then an `exp` of the same quantity, and that round trip loses accuracy
/// the division does not — the emitted result also sums to one more closely,
/// because it divides by the actual sum of the very exponentials it returns.
/// `logsoftmax` needs the `log` by definition and so has no round trip to
/// avoid; it is `shifted - log(s)`, which is `crate::ops::lower_logsumexp`'s
/// own composition with the final `+ m` and the matching `- m` cancelled
/// algebraically instead of emitted.
pub(crate) fn lower_softmax(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    which: Softmax,
) -> Result<Value, EmitError> {
    let head = if which == Softmax::Plain {
        "softmax"
    } else {
        "logsoftmax"
    };
    let [v_id] = args_exact(id, args)?;
    let v = e.lower_node(v_id)?;
    require_static_vector(id, head, &v)?;
    // §07's domain is "real vectors" here (unlike the norms' "real/complex"),
    // and `infer` types both heads a real array. §03 still admits an integer or
    // boolean operand, so widen rather than refuse.
    let v = e.convert(&v, ElemKind::Real);

    let m = e.reduce_trailing_axes(id, AxisReduce::Max, &v, 1)?;
    let m_full = splat(e, &m, &v.ty);
    let shifted = e.sub(&v, &m_full);
    let ex = e.exp(&shifted);
    let s = e.reduce_trailing_axes(id, AxisReduce::Sum, &ex, 1)?;
    match which {
        Softmax::Plain => {
            let d = splat(e, &s, &v.ty);
            Ok(e.div(&ex, &d))
        }
        Softmax::Log => {
            let log_s = e.log(&s);
            let log_s_full = splat(e, &log_s, &v.ty);
            Ok(e.sub(&shifted, &log_s_full))
        }
    }
}
