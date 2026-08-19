//! `aggregate(f_reduction, output_axes, expr)` — spec §04 "Multi-axis
//! aggregation", reached from [`crate::ops::lower_builtin`]'s `"aggregate"`
//! arm (which is also where the surface `:=` sugar lands, since
//! `flatppl_syntax` desugars `C[.i, .k] := expr` to
//! `aggregate(sum, [.i, .k], expr)` in the parser). The determiniser passes
//! `aggregate` through unchanged — FlatPDL keeps it as `aggregate`, and each
//! backend lowers it as far as it needs — so this module is where it stops.
//!
//! ## The frame model
//!
//! §04 defines the operation pointwise: `aggregate` "evaluates `expr` at every
//! combination of values of its named axes and reduces the resulting scalars by
//! `f_reduction` along the axes that do not appear in `output_axes`". FlatPPL is
//! loop-free, and so is StableHLO, so the enumeration is realized as a SHAPE:
//!
//! 1. every distinct axis name in the body gets one dimension of a common
//!    **frame**, the `output_axes` first (in their declared order) and the
//!    reduced axes after them;
//! 2. every axis-indexed operand is broadcast into that frame — a
//!    literal-index position is sliced away first, and the surviving operand
//!    axes are permuted into ascending frame order so the
//!    `stablehlo.broadcast_in_dim` needs only a monotone dimension map;
//! 3. the body is evaluated ELEMENTWISE in the frame by the ordinary
//!    [`Emitter::lower_node`] walk — each `get(A, .i, .j)` node is
//!    [`Emitter::bind`]-seeded to its frame-shaped value first, exactly the
//!    mechanism `Emitter::lower_broadcast_userfn` uses for a monomorphised
//!    `functionof` input, so no arithmetic head needs an aggregate-aware
//!    variant;
//! 4. the frame's TRAILING axes (the reduced ones) are contracted with
//!    `f_reduction` ([`Emitter::reduce_trailing_axes`]), leaving the output axes
//!    in place and in declared order — no result transpose.
//!
//! Ordering the output axes FIRST is what makes step 4 a suffix reduction and
//! removes the final permutation; nothing in §04 constrains the frame's own
//! layout, only the result's ("an array of the shape declared by
//! `output_axes`").
//!
//! `mean`/`var`/`std` are DERIVED from the sum and the (statically known)
//! reduced-element count `n`, per §07's "Reductions" table: `mean` is
//! $\frac{1}{n}\sum_i x_i$, `var` is $\frac{1}{n-1}\sum_i (x_i - \bar x)^2$
//! (Bessel-corrected — §07's row, not the population variance; §04's own
//! column-wise example prints `[32, 2, 8]` for a 2-row matrix, which is the
//! $n-1$ denominator), and `std` is $\sqrt{\mathrm{var}}$.
//!
//! ## What refuses
//!
//! The frame model cannot express a repeated axis name inside ONE index list
//! (`A[.i, .i]`, a diagonal — `stablehlo.broadcast_in_dim` requires distinct
//! dimensions), a dynamic extent (the frame's shape is static text), or a
//! chained axis index (`A[.i][.j]`). Each refuses with a located diagnostic
//! rather than lowering something else. `metricsum` refuses too, with its own
//! reason — see [`metricsum_refusal`].

use std::collections::HashMap;

use flatppl_core::{CallHead, Node, NodeId, Scalar, Symbol};

use crate::emitter::{AxisReduce, Emitter, elem_rank};
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::refuse::EmitError;

/// §04's eligible `f_reduction`s: "an order-invariant vector-to-scalar
/// reduction … The eligible built-ins are `sum`, `prod`, `mean`, `var`, `std`,
/// `maximum` and `minimum`."
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reduction {
    Sum,
    Prod,
    Mean,
    Var,
    Std,
    Max,
    Min,
}

impl Reduction {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "sum" => Reduction::Sum,
            "prod" => Reduction::Prod,
            "mean" => Reduction::Mean,
            "var" => Reduction::Var,
            "std" => Reduction::Std,
            "maximum" => Reduction::Max,
            "minimum" => Reduction::Min,
            _ => return None,
        })
    }

    /// The `stablehlo.reduce` combine this reduction contracts with —
    /// `mean`/`var`/`std` all sum, then divide.
    fn combine(self) -> AxisReduce {
        match self {
            Reduction::Sum | Reduction::Mean | Reduction::Var | Reduction::Std => AxisReduce::Sum,
            Reduction::Prod => AxisReduce::Prod,
            Reduction::Max => AxisReduce::Max,
            Reduction::Min => AxisReduce::Min,
        }
    }

    /// Whether the reduction divides by a count, so its operand must be real
    /// even when the body is integer-valued.
    fn is_moment(self) -> bool {
        matches!(self, Reduction::Mean | Reduction::Var | Reduction::Std)
    }
}

/// One index selector of an axis-indexed `get`/`get0` in the body.
enum Sel {
    /// An axis name — this operand dimension maps to that frame dimension.
    Axis(Symbol),
    /// A literal index (already 0-based) — this operand dimension is sliced
    /// away before the operand enters the frame.
    Lit(u64),
    /// §07's `only` selector (surface `B[.i, !]`), "the unique element of an
    /// axis of size 1" — §04's "Relationship to broadcasting" indexes with it,
    /// so it is aggregation-body vocabulary. Sliced away like a [`Sel::Lit`],
    /// with the length-1 requirement checked.
    Only,
}

/// One axis-indexed `get`/`get0` call in the body: the node to bind, its
/// container, and one [`Sel`] per container dimension.
struct Site {
    get_id: NodeId,
    container: NodeId,
    sels: Vec<Sel>,
}

/// The common frame every indexed operand is broadcast into: one dimension per
/// distinct axis name, the `output_axes` first (in declared order) and the
/// reduced axes after them.
struct Frame {
    axes: Vec<Symbol>,
    lens: Vec<u64>,
    /// How many leading axes are `output_axes` — the rest are reduced away.
    n_out: usize,
}

impl Frame {
    fn ty(&self) -> MlirTy {
        MlirTy::Ranked(self.lens.iter().map(|&n| Some(n)).collect())
    }

    fn pos(&self, axis: Symbol) -> Option<usize> {
        self.axes.iter().position(|&a| a == axis)
    }

    /// The number of body evaluations each output cell reduces over — the
    /// product of the reduced axes' lengths (`1` when nothing is reduced).
    fn reduced_count(&self) -> u64 {
        self.lens[self.n_out..].iter().product()
    }
}

/// Lower `aggregate(f_reduction, output_axes, expr)` (spec §04) — see the
/// module doc for the frame model.
pub(crate) fn lower_aggregate(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [f_id, axes_id, body] = <[NodeId; 3]>::try_from(args).map_err(|_| {
        EmitError::at(
            id,
            format!(
                "aggregate: expected 3 arguments (f_reduction, output_axes, expr), got {}",
                args.len()
            ),
        )
    })?;

    let reduction = read_reduction(e, f_id)?;
    let output_axes = read_output_axes(e, axes_id)?;

    let mut sites = Vec::new();
    collect_sites(e, body, &mut sites)?;

    let frame = build_frame(e, id, &output_axes, &sites)?;

    // Bound WITHOUT snapshotting the body subtree's memo, unlike the crate's
    // other `bind`-seeding site (`Emitter::lower_broadcast_userfn`, which restores
    // it so one `functionof` re-lowers freshly under different arguments): a
    // frame-shaped value here can never be read under a different frame, because
    // `flatppl-core` interns NAMES only and hash-conses no nodes, so two
    // aggregates over identical body text still hold distinct `NodeId`s — and an
    // axis index cannot occur in a top-level binding, so no `Ref` can share one
    // across aggregates either.
    for site in &sites {
        let v = frame_operand(e, site, &frame)?;
        e.bind(site.get_id, v);
    }

    let cell = e.lower_node(body)?;
    let cell = require_frame_shaped(e, id, &cell, &frame)?;
    reduce(e, id, reduction, &cell, &frame)
}

/// The refusal for `metricsum(metric, output_axes, expr)` (spec §04
/// "Metric-aware Einstein summation"), which this backend does NOT lower — as a
/// CONSTRUCT, whatever variances one particular call happens to use.
///
/// The blocker is the general case. §04 "Lowering to `aggregate`" turns each
/// lower-variance axis into an `inv(metric)` contraction, so a metricsum needs
/// the INVERSE of an arbitrary square symmetric invertible matrix — §04 requires
/// only that the metric be "square, symmetric, and invertible", not
/// positive-definite, and the section's own worked example is a Lorentz metric,
/// which is indefinite. StableHLO has no matrix-inverse or LU op, and
/// `stablehlo.cholesky` (this crate's only factorization) needs
/// positive-definiteness, so the missing piece is a general indefinite inverse —
/// different machinery from the frame model above, not a missing arm of it.
///
/// The all-upper degenerate case (`g: r[.mu^] := v[.mu^]`) needs no inverse and
/// would reduce to a plain `aggregate(sum, …)`, but it is refused with everything
/// else rather than carved out: supporting it would still need §04's
/// metricsum-only static checks (every repeated non-output index exactly twice,
/// once upper and once lower; no bare neutral axes), and a backend that accepted
/// some variance patterns and refused others would be a worse contract than one
/// that declines the construct. The message therefore states what a metricsum
/// needs IN GENERAL, and names the rewrite that works for every case including
/// the degenerate one. Refused here rather than left to `ops::lower_builtin`'s
/// generic "unsupported builtin head".
pub(crate) fn metricsum_refusal(id: NodeId) -> EmitError {
    EmitError::at(
        id,
        "metricsum has no lowering in this backend: in general §04 \"Lowering to `aggregate`\" \
         makes each lower-variance axis an `inv(metric)` contraction, and §04 requires the metric \
         only to be \"square, symmetric, and invertible\" — not positive-definite — so it needs a \
         general indefinite matrix inverse, which StableHLO has no op for (`stablehlo.cholesky` \
         requires positive-definiteness). The construct is declined as a whole rather than for \
         some variance patterns only, so this call is refused even if its own indices need no \
         inverse. Contract any metric factors explicitly and write the result as \
         `aggregate(sum, …)`, which does lower",
    )
}

/// Read the `f_reduction` argument. §04 admits only the seven built-in
/// order-invariant reductions, named bare — a user function or a reference to a
/// binding refuses rather than being called elementwise.
fn read_reduction(e: &Emitter, f_id: NodeId) -> Result<Reduction, EmitError> {
    let name = match e.node(f_id) {
        Node::Const(sym) => e.resolve(*sym),
        _ => {
            return Err(EmitError::at(
                f_id,
                "aggregate: f_reduction must be one of the built-in reductions named bare",
            ));
        }
    };
    Reduction::parse(name).ok_or_else(|| {
        EmitError::at(
            f_id,
            format!(
                "aggregate: `{name}` is not an eligible reduction — §04 \"Multi-axis \
                 aggregation\" requires \"an order-invariant vector-to-scalar reduction\" and \
                 lists the eligible built-ins as `sum`, `prod`, `mean`, `var`, `std`, `maximum` \
                 and `minimum`"
            ),
        )
    })
}

/// Read `output_axes` — §04's "axis list of distinct axis names `[.name1,
/// .name2, ...]` listing the retained axes in output order. Repeated names are
/// a static error. The empty axis list `[]` is legal and denotes full reduction
/// to a scalar."
///
/// The list is the `vector(...)` call the parser builds for the bracket
/// literal; anything else (a computed value, a vector of non-axes) refuses,
/// since the result RANK would not be statically known.
fn read_output_axes(e: &Emitter, axes_id: NodeId) -> Result<Vec<Symbol>, EmitError> {
    let refuse = || {
        EmitError::at(
            axes_id,
            "aggregate: output_axes must be a literal axis list `[.i, .k]` (possibly empty)",
        )
    };
    let Node::Call(c) = e.node(axes_id) else {
        return Err(refuse());
    };
    if !matches!(c.head, CallHead::Builtin(sym) if e.resolve(sym) == "vector") {
        return Err(refuse());
    }
    let mut out: Vec<Symbol> = Vec::with_capacity(c.args.len());
    for &a in c.args.iter() {
        let Node::Axis(ax) = e.node(a) else {
            return Err(refuse());
        };
        if ax.variance.is_some() {
            return Err(EmitError::at(
                a,
                format!(
                    "aggregate: `.{}` carries a variance marker, which §05 \"Axis names and \
                     aggregation\" admits only inside `metricsum`",
                    e.resolve(ax.name)
                ),
            ));
        }
        if out.contains(&ax.name) {
            return Err(EmitError::at(
                a,
                format!(
                    "aggregate: output axis `.{}` is repeated — §04 requires output_axes to be \
                     distinct axis names",
                    e.resolve(ax.name)
                ),
            ));
        }
        out.push(ax.name);
    }
    Ok(out)
}

/// Collect the body's axis-indexed `get`/`get0` calls, in encounter order.
///
/// Does NOT descend into a nested `aggregate`/`metricsum`: §04 makes axis names
/// "lexically scoped to the enclosing `aggregate(...)`", so an inner
/// aggregation's axes are its own and its node lowers through the ordinary
/// [`Emitter::lower_node`] dispatch (re-entering this module) when the body is
/// walked.
fn collect_sites(e: &Emitter, node: NodeId, out: &mut Vec<Site>) -> Result<(), EmitError> {
    let Node::Call(c) = e.node(node) else {
        return Ok(());
    };
    let head = match c.head {
        CallHead::Builtin(sym) => e.resolve(sym),
        CallHead::User(_) => "",
    };
    if matches!(head, "aggregate" | "metricsum") {
        return Ok(());
    }
    let indexing = matches!(head, "get" | "get0")
        && c.args
            .iter()
            .skip(1)
            .any(|&a| matches!(e.node(a), Node::Axis(_)));
    if indexing {
        let base: u64 = if head == "get" { 1 } else { 0 };
        let container = c.args[0];
        // A chained axis index (`A[.i][.j]`) would need this operand's own frame
        // value before the frame exists — refuse rather than order it wrongly.
        let mut inner = Vec::new();
        collect_sites(e, container, &mut inner)?;
        if !inner.is_empty() {
            return Err(EmitError::at(
                node,
                "aggregate: chained axis indexing (`A[.i][.j]`) has no frame form — write the \
                 axes in one index list (`A[.i, .j]`)",
            ));
        }
        let mut sels: Vec<Sel> = Vec::with_capacity(c.args.len() - 1);
        for &sel in c.args.iter().skip(1) {
            match e.node(sel) {
                Node::Axis(ax) => {
                    if ax.variance.is_some() {
                        return Err(EmitError::at(
                            sel,
                            format!(
                                "aggregate: `.{}` carries a variance marker, which §05 \"Axis \
                                 names and aggregation\" admits only inside `metricsum`",
                                e.resolve(ax.name)
                            ),
                        ));
                    }
                    if sels
                        .iter()
                        .any(|s| matches!(s, Sel::Axis(n) if *n == ax.name))
                    {
                        return Err(EmitError::at(
                            node,
                            format!(
                                "aggregate: axis `.{}` indexes two dimensions of one operand, \
                                 which denotes a diagonal — the aggregation frame gives each axis \
                                 name one dimension, so it has no form for this",
                                e.resolve(ax.name)
                            ),
                        ));
                    }
                    sels.push(Sel::Axis(ax.name));
                }
                Node::Lit(Scalar::Int(i)) => {
                    let idx = *i - base as i64;
                    if idx < 0 {
                        return Err(EmitError::at(
                            sel,
                            format!("aggregate: index {i} is out of range"),
                        ));
                    }
                    sels.push(Sel::Lit(idx as u64));
                }
                Node::Const(sym) if e.resolve(*sym) == "only" => sels.push(Sel::Only),
                Node::Const(sym) if e.resolve(*sym) == "all" => {
                    return Err(EmitError::at(
                        sel,
                        "aggregate: the `all` selector selects a whole axis (§07 \"Axis slicing \
                         with `all`\"), so the body would evaluate to an array per axis \
                         combination — §04 reduces \"the resulting scalars\". Name the axis \
                         instead (`A[.i, .j]`)",
                    ));
                }
                _ => {
                    return Err(EmitError::at(
                        sel,
                        "aggregate: an index in an aggregation body must be an axis name, an \
                         integer literal, or `!` (§04 admits `A[.i, 1, .j]` and `B[.i, !]`)",
                    ));
                }
            }
        }
        out.push(Site {
            get_id: node,
            container,
            sels,
        });
        return Ok(());
    }
    for &a in c.args.iter() {
        collect_sites(e, a, out)?;
    }
    for named in c.named.iter() {
        collect_sites(e, named.value, out)?;
    }
    Ok(())
}

/// Lower each site's container, check its shape against the site's selectors,
/// and assemble the frame: the `output_axes` in declared order, then the
/// reduced axes in first-encounter order.
///
/// §04: "All array dimensions indexed with the same axis name must have the same
/// length" (a disagreement is a static error), and "Every axis name in
/// `output_axes` must occur at least once in `expr`".
fn build_frame(
    e: &mut Emitter,
    id: NodeId,
    output_axes: &[Symbol],
    sites: &[Site],
) -> Result<Frame, EmitError> {
    // Axis extents, in first-encounter order over the sites.
    let mut lens: HashMap<Symbol, u64> = HashMap::new();
    let mut seen: Vec<Symbol> = Vec::new();
    for site in sites.iter() {
        let container = e.lower_node(site.container)?;
        let dims = match &container.ty {
            MlirTy::Ranked(dims) => dims.clone(),
            other => {
                return Err(EmitError::at(
                    site.get_id,
                    format!(
                        "aggregate: an axis-indexed operand must be an array, got {other:?} — \
                         §04 indexes `expr`'s operands with axis names, so a non-tensor operand \
                         has no frame form"
                    ),
                ));
            }
        };
        if dims.len() != site.sels.len() {
            return Err(EmitError::at(
                site.get_id,
                format!(
                    "aggregate: {} index selector(s) for a rank-{} operand — a partial index \
                     yields an array per axis combination, and §04 reduces \"the resulting \
                     scalars\", so every dimension must be selected (§04's `A[.i, 1, .j]` names \
                     all three)",
                    site.sels.len(),
                    dims.len()
                ),
            ));
        }
        for (k, sel) in site.sels.iter().enumerate() {
            let Some(len) = dims[k] else {
                return Err(EmitError::at(
                    site.get_id,
                    "aggregate: a dynamic (`?`) dimension has no aggregation frame — the frame's \
                     shape is static text, and §04's equal-length rule cannot be checked against \
                     an unknown extent",
                ));
            };
            match sel {
                Sel::Lit(idx) => {
                    if *idx >= len {
                        return Err(EmitError::at(
                            site.get_id,
                            format!(
                                "aggregate: 0-based index {idx} is out of range for a dimension \
                                 of length {len}"
                            ),
                        ));
                    }
                }
                Sel::Only => {
                    if len != 1 {
                        return Err(EmitError::at(
                            site.get_id,
                            format!(
                                "aggregate: `!` indexes a dimension of length {len} — §07 \
                                 \"Singleton-axis indexing with `only`\" requires that \"the \
                                 indexed axis must be of length one\""
                            ),
                        ));
                    }
                }
                Sel::Axis(name) => match lens.get(name) {
                    Some(&prev) if prev != len => {
                        return Err(EmitError::at(
                            site.get_id,
                            format!(
                                "aggregate: axis `.{}` indexes dimensions of different lengths, \
                                 {prev} and {len} — §04 requires that \"all array dimensions \
                                 indexed with the same axis name must have the same length\"",
                                e.resolve(*name)
                            ),
                        ));
                    }
                    Some(_) => {}
                    None => {
                        lens.insert(*name, len);
                        seen.push(*name);
                    }
                },
            }
        }
    }

    let mut axes: Vec<Symbol> = Vec::with_capacity(seen.len());
    for &out in output_axes {
        if !lens.contains_key(&out) {
            return Err(EmitError::at(
                id,
                format!(
                    "aggregate: output axis `.{}` does not index anything in the body — §04 \
                     requires that \"every axis name in `output_axes` must occur at least once in \
                     `expr`\"",
                    e.resolve(out)
                ),
            ));
        }
        axes.push(out);
    }
    let n_out = axes.len();
    for name in seen {
        if !axes.contains(&name) {
            axes.push(name);
        }
    }
    let lens: Vec<u64> = axes.iter().map(|a| lens[a]).collect();
    Ok(Frame { axes, lens, n_out })
}

/// Bring one site's operand into the frame: slice away its literal-index
/// dimensions, permute the surviving ones into ascending frame order, then
/// `broadcast_in_dim` up to the frame's shape.
///
/// The permutation is emitted as its own `stablehlo.transpose` rather than
/// folded into `broadcast_in_dim`'s dimension map: StableHLO's
/// `broadcast_dimensions` need only be unique, but a NON-MONOTONE map is a
/// transposing broadcast that consumers vary in accepting, so the two steps stay
/// separate and the emitted map is always increasing.
fn frame_operand(e: &mut Emitter, site: &Site, frame: &Frame) -> Result<Value, EmitError> {
    let operand = e.lower_node(site.container)?;
    let MlirTy::Ranked(dims) = operand.ty.clone() else {
        // `build_frame` already refused a non-ranked operand.
        return Err(EmitError::at(
            site.get_id,
            "aggregate: operand is not ranked",
        ));
    };

    // 1. Drop the literal-index and `only` dimensions.
    let has_lit = site
        .sels
        .iter()
        .any(|s| matches!(s, Sel::Lit(_) | Sel::Only));
    let sliced = if has_lit {
        let mut starts = Vec::with_capacity(dims.len());
        let mut limits = Vec::with_capacity(dims.len());
        let mut kept: Vec<Option<u64>> = Vec::new();
        for (k, sel) in site.sels.iter().enumerate() {
            let len = dims[k].expect("static extent checked in build_frame");
            match sel {
                Sel::Lit(idx) => {
                    starts.push(*idx);
                    limits.push(idx + 1);
                }
                Sel::Only => {
                    starts.push(0);
                    limits.push(1);
                }
                Sel::Axis(_) => {
                    starts.push(0);
                    limits.push(len);
                    kept.push(Some(len));
                }
            }
        }
        // A slice that keeps every element (every dropped dimension already has
        // length 1) is emitted as nothing — only the reshape is needed.
        let trivial =
            starts.iter().all(|&s| s == 0) && limits.iter().zip(&dims).all(|(&l, d)| Some(l) == *d);
        let v = if trivial {
            operand
        } else {
            let strides = vec![1u64; dims.len()];
            e.slice(&operand, &starts, &limits, &strides)
        };
        let kept_ty = if kept.is_empty() {
            MlirTy::Scalar
        } else {
            MlirTy::Ranked(kept)
        };
        e.reshape(&v, kept_ty)
    } else {
        operand
    };

    // 2. The frame position of each surviving operand dimension, in order.
    let positions: Vec<usize> = site
        .sels
        .iter()
        .filter_map(|s| match s {
            Sel::Axis(name) => frame.pos(*name),
            Sel::Lit(_) | Sel::Only => None,
        })
        .collect();

    // 3. Permute them into ascending frame order.
    let mut order: Vec<usize> = (0..positions.len()).collect();
    order.sort_by_key(|&j| positions[j]);
    let permuted = if order.windows(2).all(|w| w[0] < w[1]) {
        sliced
    } else {
        let perm: Vec<u64> = order.iter().map(|&j| j as u64).collect();
        e.transpose(&sliced, &perm)
    };
    let mut dims_map: Vec<u64> = order.iter().map(|&j| positions[j] as u64).collect();
    dims_map.sort_unstable();

    // 4. Broadcast up to the frame.
    let frame_ty = frame.ty();
    if permuted.ty == frame_ty && dims_map.len() == frame.lens.len() {
        return Ok(permuted);
    }
    Ok(e.broadcast_in_dim(&permuted, &dims_map, frame_ty))
}

/// The body's value must have the frame's exact shape — every axis-indexed
/// operand was broadcast to the full frame, so an elementwise body is
/// frame-shaped by construction. A body that changed the rank (a `sum(...)`
/// of a frame tensor, say) refuses rather than being reduced against a shape it
/// does not have.
///
/// A SCALAR body is accepted and broadcast up: the frame is then either empty
/// (nothing in the body is axis-indexed) or the body dropped its indexed
/// operands, and either way every cell holds the same value.
fn require_frame_shaped(
    e: &mut Emitter,
    id: NodeId,
    cell: &Value,
    frame: &Frame,
) -> Result<Value, EmitError> {
    if frame.lens.is_empty() {
        return Ok(cell.clone());
    }
    let frame_ty = frame.ty();
    if cell.ty == frame_ty {
        return Ok(cell.clone());
    }
    if cell.ty == MlirTy::Scalar {
        return Ok(e.broadcast_in_dim(cell, &[], frame_ty));
    }
    Err(EmitError::at(
        id,
        format!(
            "aggregate: the body evaluates to {:?}, not to the aggregation frame's {frame_ty:?} \
             — §04 requires `expr` to produce one value per combination of its axes, so a body \
             that reshapes or reduces its indexed operands has no frame form",
            cell.ty
        ),
    ))
}

/// Contract the frame's reduced (trailing) axes with `reduction`.
///
/// `sum`/`prod`/`maximum`/`minimum` are one [`Emitter::reduce_trailing_axes`]
/// call. `mean`/`var`/`std` are derived from the sum and the statically known
/// reduced-element count `n`, exactly as §07's "Reductions" table defines them
/// — `var` with the $n-1$ denominator, so it needs `n >= 2`.
fn reduce(
    e: &mut Emitter,
    id: NodeId,
    reduction: Reduction,
    cell: &Value,
    frame: &Frame,
) -> Result<Value, EmitError> {
    let n_red = frame.lens.len() - frame.n_out;
    let count = frame.reduced_count();

    // The result must carry the element kind inference recorded for this node
    // (`Real` whenever the body's own type is `%deferred`, which is the common
    // case for an axis-indexed body) — widen the cell first so the reduction
    // runs in that kind. Only ever a widening: §03's `booleans ⊂ integers ⊂
    // reals` embedding is exact, a narrowing convert would not be.
    let target = e.node_kind(id);
    let cell = if elem_rank(cell.elem) < elem_rank(target) {
        e.convert(cell, target)
    } else {
        cell.clone()
    };
    // A moment divides by a count, which is real arithmetic whatever the body's
    // kind.
    let cell = if reduction.is_moment() {
        e.convert(&cell, ElemKind::Real)
    } else {
        cell
    };

    let total = e.reduce_trailing_axes(id, reduction.combine(), &cell, n_red)?;
    if !reduction.is_moment() {
        return Ok(total);
    }

    if count == 0 {
        return Err(EmitError::at(
            id,
            format!(
                "aggregate: `{}` over an empty axis is undefined — §07 divides by the element \
                 count, which is zero here",
                reduction_name(reduction)
            ),
        ));
    }
    let n = e.scalar(count as f64);
    let mean = e.div(&total, &n);
    if reduction == Reduction::Mean {
        return Ok(mean);
    }

    if count < 2 {
        return Err(EmitError::at(
            id,
            format!(
                "aggregate: `{}` over {count} element(s) is undefined — §07 defines it with the \
                 $n-1$ denominator, so it needs at least two elements per output cell",
                reduction_name(reduction)
            ),
        ));
    }
    // The centred sum of squares, per output cell: the mean broadcasts back over
    // the reduced axes under the identity map (it carries the frame's leading
    // `n_out` dimensions, so the map is `[0, 1, …, n_out - 1]`).
    let dims: Vec<u64> = (0..frame.n_out as u64).collect();
    let mean_frame = e.broadcast_in_dim(&mean, &dims, frame.ty());
    let dev = e.sub(&cell, &mean_frame);
    let sq = e.mul(&dev, &dev);
    let ssq = e.reduce_trailing_axes(id, AxisReduce::Sum, &sq, n_red)?;
    let denom = e.scalar((count - 1) as f64);
    let var = e.div(&ssq, &denom);
    if reduction == Reduction::Var {
        return Ok(var);
    }
    Ok(e.sqrt(&var))
}

/// The §04 spelling of a reduction, for a refusal message.
fn reduction_name(r: Reduction) -> &'static str {
    match r {
        Reduction::Sum => "sum",
        Reduction::Prod => "prod",
        Reduction::Mean => "mean",
        Reduction::Var => "var",
        Reduction::Std => "std",
        Reduction::Max => "maximum",
        Reduction::Min => "minimum",
    }
}
