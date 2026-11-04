//! `markovchain(kernel, init, n)` / `kscan(kernel, init, xs)` density lowering
//! (spec §06 "Dependent composition").
//!
//! Both are trajectory measures. §06 `markovchain`: "Step $i$ is
//! $\text{traj}_i \sim \kappa(\text{traj}_{i-1})$ with
//! $\text{traj}_0 = \text{init}$. The initial value is not part of the
//! trajectory." §06 `kscan`: "step $i$ is
//! $\text{traj}_i \sim \kappa(\text{traj}_{i-1}, \text{xs}_i)$ with
//! $\text{traj}_0 = \text{init}$", trajectory length `lengthof(xs)`.
//!
//! Each step is one Markov kernel applied to the previous state, so the
//! trajectory is a dependent product with no marginalization — the same shape
//! §06 "Density of composed measures" gives `jointchain`: "the product of the
//! constituent conditional densities":
//!
//! ```text
//! logdensityof(markovchain(K, init, n), v)
//!   = Σ_{i<n} logdensityof(K(prevᵢ), vᵢ)
//!
//! logdensityof(kscan(K, init, xs), v)
//!   = Σ_{i<n} logdensityof(K(prevᵢ, xsᵢ), vᵢ),   prev = cat([init], v[0..n-2])
//! ```
//!
//! `init` carries no density: it is a VALUE in the state space, not a draw, and
//! §06 excludes it from the trajectory. So the sum has exactly `n` terms, one
//! per trajectory element — never `n + 1`.
//!
//! **One mapped body, no loop.** The point `v` is observed, so term `i` reads only
//! the known `prevᵢ` and `vᵢ`, and no term depends on another. The lowering is
//! `sum(broadcast(body, prev, v, xs...))`, one density body mapped over the step
//! axis, as [`crate::density::lower_iid`] maps its composed `M`. Its size does
//! not grow with `n`. An unroll grows the program, and its compile time grows
//! faster than linearly; a `while` loop runs the steps in sequence although
//! they are independent.
//!
//! The trajectory length is read from the chain node's own inferred domain shape
//! (`flatppl_infer::trajectory_measure` builds `array[len]` of `init`'s type,
//! folding `markovchain`'s `n` at `Level::Shape` and taking `kscan`'s from `xs`'s
//! leading dim). A dynamic length refuses, since the shifted array needs a
//! static extent.
//!
//! The step is emitted as a kernel APPLICATION `K(prev)` / `K(prev, x)` handed
//! back to the density dispatcher, rather than substituted here. That reuses the
//! applied-reification path's boundary guards (`substitute_applied_boundary`'s
//! through-binding finish and residual check) instead of a second, weaker copy
//! of them, and it lets a composed step body (`logweighted(ℓ, M)`, `pushfwd`, …)
//! lower through its own combinator rule.

use crate::density::broadcast::{cell_input, map_body, placeholder};
use crate::density::{build_call, lower_measure_density_at_point, refuse, trajectory_static_len};
use crate::kernel::resolve_reified;
use crate::refuse::RefuseError;
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, Type};

/// Lower `logdensityof(markovchain(K, init, n), v)`.
pub(crate) fn lower_markovchain(
    m: &mut Module,
    node: NodeId,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    lower_chain(m, node, v, "markovchain")
}

/// Lower `logdensityof(kscan(K, init, xs), v)`.
pub(crate) fn lower_kscan(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    lower_chain(m, node, v, "kscan")
}

/// The shared trajectory unroll. `op` is `"markovchain"` (kernel, init, `n`, then
/// additional arguments held constant) or `"kscan"` (kernel, init, then sequences
/// traversed together, spec §04 "Additional arguments").
fn lower_chain(m: &mut Module, node: NodeId, v: NodeId, op: &str) -> Result<NodeId, RefuseError> {
    let per_step_input = op == "kscan";
    let lead = if per_step_input { 2 } else { 3 };
    let (k_arg, init) = {
        let Node::Call(c) = m.node(node) else {
            return Err(refuse(node, m, "expected a chain call"));
        };
        if c.args.len() < lead {
            return Err(refuse(
                node,
                m,
                &format!(
                    "{op} expects (kernel, init, {})",
                    if per_step_input {
                        "sequences..."
                    } else {
                        "n, additional arguments..."
                    }
                ),
            ));
        }
        (c.args[0], c.args[1])
    };
    let extras = crate::extras::bind_extras(m, node, k_arg, lead, 1)?;
    let expected_inputs = 1 + extras.len();

    // The step count. `None` covers a dynamic length AND a record-state chain
    // (whose trajectory is a table, left with a deferred domain by inference).
    let n = trajectory_static_len(m, node).ok_or_else(|| {
        refuse(
            node,
            m,
            &format!(
                "{op} trajectory length is not a statically-resolved 1-D count \
                 (a dynamic length, or a record-state chain whose trajectory is a \
                 table): only a static-length array trajectory is lowered"
            ),
        )
    })?;

    // The kernel's arity must match the §06 signature — `(state)` for
    // `markovchain`, `(state, x)` for `kscan` — and the boundary must be
    // `%specinputs`, since the steps bind it BY POSITION. An `%autoinputs`
    // boundary is keyword-only (§04: "no argument order can be inferred"), so
    // positional binding of one would attach the state to an arbitrarily
    // ordered traced input.
    let kernel = resolve_reified(m, k_arg).ok_or_else(|| {
        refuse(
            node,
            m,
            &format!("{op} kernel is not a functionof/kernelof with boundary inputs"),
        )
    })?;
    if kernel.auto {
        return Err(refuse(
            node,
            m,
            &format!(
                "{op} kernel has an `%autoinputs` (keyword-only) boundary, but §06 \
                 gives the step kernel a POSITIONAL signature; spell the boundary \
                 explicitly or write the kernel as a lambda"
            ),
        ));
    }
    if kernel.inputs.len() != expected_inputs {
        return Err(refuse(
            node,
            m,
            &format!(
                "{op} kernel takes {} boundary input(s), but §06 gives it {expected_inputs}",
                kernel.inputs.len()
            ),
        ));
    }

    // Empty trajectory: the product over an empty index set is 1, so the
    // log-density is 0 — the same empty-Σ rule §06 "Density of composed
    // measures" states for `iid`. The map needs a non-empty step axis.
    if n == 0 {
        return Ok(m.alloc(Node::Lit(Scalar::Real(0.0))));
    }

    // §04 "Additional arguments": a one-dimensional array or table is traversed with
    // the steps, a length-one one repeats, and anything else is held constant.
    let positions: Vec<Option<usize>> = if per_step_input {
        extras
            .iter()
            .map(|&extra| match extra_traversal(m, extra) {
                Some(Some(1)) => Ok(Some(1)),
                Some(Some(len)) if len == n => Ok(Some(len)),
                Some(None) => Ok(None),
                _ => Err(refuse(
                    node,
                    m,
                    "kscan additional input is not statically a constant or a sequence of the \
                     trajectory length",
                )),
            })
            .collect::<Result<_, _>>()?
    } else {
        vec![None; extras.len()]
    };

    // Term i reads only the observed `v[i-1]` and `v[i]`, so one body maps over
    // `prev = cat([init], v[0..n-2])` and `cur = v`.
    let state = match m.type_of(node) {
        Some(Type::Measure { domain, .. }) => match domain.as_ref() {
            Type::Array { elem, .. } => (**elem).clone(),
            _ => unreachable!("trajectory_static_len read an array domain"),
        },
        _ => unreachable!("trajectory_static_len read a measure type"),
    };
    let (prev_label, prev_ref, prev) = placeholder(m, state.clone());
    let (cur_label, cur_ref, cur) = placeholder(m, state);
    let mut inputs = vec![(prev_label, prev_ref), (cur_label, cur_ref)];
    // Inside an enclosing map `v` is a bare cell, whose kind a static `cat`
    // check cannot see. A selection from it is visibly a vector.
    let head = build_call(m, "vector", &[init]);
    let shifted = if n == 1 {
        head
    } else {
        let count = m.alloc(Node::Lit(Scalar::Int(n as i64 - 1)));
        let zeros = build_call(m, "zeros", &[count]);
        let indices = build_call(m, "indicesof0", &[zeros]);
        let leading = build_call(m, "get0", &[v, indices]);
        build_call(m, "cat", &[head, leading])
    };
    let mut args = vec![shifted, v];
    let mut step_extras = Vec::with_capacity(extras.len());
    for (&extra, &position) in extras.iter().zip(&positions) {
        step_extras.push(match position {
            Some(1) => {
                let idx = m.alloc(Node::Lit(Scalar::Int(0)));
                build_call(m, "get0", &[extra, idx])
            }
            Some(_) => cell_input(m, extra, &mut inputs, &mut args)?,
            None => extra,
        });
    }
    let step = crate::extras::apply_callback(m, k_arg, prev, &step_extras)?;
    // The cell IS this step measure's own variate, supplied by the point.
    let body = lower_measure_density_at_point(m, step, cur)?;
    let terms = map_body(m, body, inputs, args, 1);
    Ok(build_call(m, "sum", &[terms]))
}

/// How an additional input of `kscan` meets the steps: `Some(Some(len))` for a
/// one-dimensional array or table of static length `len`, `Some(None)` for a
/// constant, and `None` when its type is not statically known.
fn extra_traversal(m: &mut Module, extra: NodeId) -> Option<Option<usize>> {
    let resolved = crate::density::resolve_ref_chain(m, extra).0;
    let ty = m
        .type_of(extra)
        .or_else(|| m.type_of(resolved))
        .cloned()
        .or_else(|| {
            flatppl_infer::infer_expression(m, extra, &[])
                .0
                .remove(&extra)
        })?;
    let dim = match ty {
        Type::Array { shape, .. } if shape.len() == 1 => shape[0],
        Type::Table { nrows, .. } => nrows,
        Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => return None,
        _ => return Some(None),
    };
    match dim {
        Dim::Static(n) => Some(Some(n as usize)),
        Dim::Dynamic => None,
    }
}
