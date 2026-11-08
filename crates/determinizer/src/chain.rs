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

use crate::density::broadcast::{Lift, cell_input, closed_function, map_body, placeholder};
use crate::density::{
    build_call, build_record, lower_measure_density_at_point, refuse, trajectory_static_len,
};
use crate::kernel::resolve_reified;
use crate::refuse::RefuseError;
use flatppl_core::{Binding, Dim, Module, Node, NodeId, Ref, RefNs, Scalar, ScalarType, Type};

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

/// A chain call's checked arguments: `op` is `"markovchain"` (kernel, init, `n`,
/// then an optional parameter `p`) or `"kscan"` (kernel, init, the sequence `xs`,
/// then an optional `p`). `p` is passed whole to every step (§04 "Explicit
/// callback parameters").
struct ChainParts {
    k_arg: NodeId,
    init: NodeId,
    extras: Vec<NodeId>,
    /// Per extra: `Some(len)` traversed with the steps, `None` passed whole.
    positions: Vec<Option<usize>>,
    n: usize,
    state: Type,
}

fn chain_parts(m: &mut Module, node: NodeId, op: &str) -> Result<ChainParts, RefuseError> {
    let per_step_input = op == "kscan";
    // `kscan`'s `xs` is a leading argument like `markovchain`'s `n`; only `p` binds
    // through the callback-parameter rule.
    let lead = 3;
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
                    if per_step_input { "xs[, p]" } else { "n[, p]" }
                ),
            ));
        }
        (c.args[0], c.args[1])
    };
    let xs = per_step_input.then(|| match m.node(node) {
        Node::Call(c) => c.args[2],
        _ => unreachable!("checked above"),
    });
    let callback_lead = if per_step_input { 2 } else { 1 };
    let parameters = crate::extras::bind_extras(m, node, k_arg, lead, callback_lead)?;
    let extras: Vec<NodeId> = xs.into_iter().chain(parameters).collect();
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

    // `xs` is traversed with the steps; a final `p` is passed whole, whatever its
    // shape (§04 "Explicit callback parameters").
    let mut positions: Vec<Option<usize>> = vec![None; extras.len()];
    if per_step_input && let Some(&xs) = extras.first() {
        positions[0] = match extra_traversal(m, xs) {
            Some(Some(len)) if len == n => Some(len),
            _ => {
                return Err(refuse(
                    node,
                    m,
                    "kscan sequence `xs` is not statically a one-dimensional array or table of \
                     the trajectory length",
                ));
            }
        };
    }

    let state = match m.type_of(node) {
        Some(Type::Measure { domain, .. }) => match domain.as_ref() {
            Type::Array { elem, .. } => (**elem).clone(),
            _ => unreachable!("trajectory_static_len read an array domain"),
        },
        _ => unreachable!("trajectory_static_len read a measure type"),
    };
    Ok(ChainParts {
        k_arg,
        init,
        extras,
        positions,
        n,
        state,
    })
}

/// The shared trajectory density.
fn lower_chain(m: &mut Module, node: NodeId, v: NodeId, op: &str) -> Result<NodeId, RefuseError> {
    let ChainParts {
        k_arg,
        init,
        extras,
        positions,
        n,
        state,
    } = chain_parts(m, node, op)?;
    // Empty trajectory: the product over an empty index set is 1, so the
    // log-density is 0 — the same empty-Σ rule §06 "Density of composed
    // measures" states for `iid`. The map needs a non-empty step axis.
    if n == 0 {
        return Ok(m.alloc(Node::Lit(Scalar::Real(0.0))));
    }

    // Term i reads only the observed `v[i-1]` and `v[i]`, so one body maps over
    // `prev = cat([init], v[0..n-2])` and `cur = v`.
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

/// The static length of `kscan`'s sequence `xs`: `Some(Some(len))` for a
/// one-dimensional array or table, `Some(None)` for anything else, and `None` when
/// its type is not statically known.
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

/// Sample `markovchain(K, init, n)` / `kscan(K, init, xs)` from `rng`
/// (spec §06 "Dependent composition", §07 "Random number generation"):
///
/// ```text
/// traj = scan(c, x -> record(state = v, rng = r'), record(state = init, rng = rng), xs)
///   where v, r' = sample of K(c.state, x) from c.rng
/// value = traj.state,   next rng = get0(traj.rng, n - 1)
/// ```
///
/// Step `i` draws from the state the previous step drew, so the steps run in
/// sequence and the rng state rides in the carry. `init` is not part of the
/// trajectory, so the scan's output column is exactly the trajectory.
pub(crate) fn lower_chain_sample(
    m: &mut Module,
    node: NodeId,
    rng: NodeId,
    op: &str,
) -> Result<(NodeId, NodeId), RefuseError> {
    let ChainParts {
        k_arg,
        init,
        extras,
        positions,
        n,
        state,
    } = chain_parts(m, node, op)?;
    if n == 0 {
        return Err(refuse(
            node,
            m,
            &format!("{op} sample: an empty trajectory is not lowered"),
        ));
    }
    let state_name = m.intern("state");
    let rng_name = m.intern("rng");
    let carry_ty =
        Type::Record(vec![(state_name, state.clone()), (rng_name, Type::RngState)].into());
    let (carry_label, carry_ref, carry) = placeholder(m, carry_ty);
    let mut inputs = vec![(carry_label, carry_ref)];
    let mut args = Vec::new();
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
    // `scan(f, init, xs, p)` traverses one sequence and passes `p` whole
    // (spec §04 "Explicit callback parameters"). `markovchain` has no
    // sequence, so its steps walk a vector the step does not read.
    if args.len() > 1 {
        return Err(refuse(
            node,
            m,
            &format!("{op} sample: more than one sequence is traversed; kscan takes one"),
        ));
    }
    if args.is_empty() {
        let (label, reference, _) = placeholder(m, Type::Scalar(ScalarType::Real));
        inputs.push((label, reference));
        let count = m.alloc(Node::Lit(Scalar::Int(n as i64)));
        args.push(build_call(m, "zeros", &[count]));
    }
    let prev = field(m, carry, "state");
    m.set_type(prev, state);
    let current = field(m, carry, "rng");
    m.set_type(current, Type::RngState);
    let step = crate::extras::apply_callback(m, k_arg, prev, &step_extras)?;
    let (value, next) = crate::sample::lower_closed_measure_sample(m, step, current)?;
    let body = build_record(m, &[(state_name, value), (rng_name, next)]);
    let (function, sequences) = closed_function(m, body, inputs, args, Lift::Parameter);
    let start = build_record(m, &[(state_name, init), (rng_name, rng)]);
    let mut scan_args = vec![function, start];
    scan_args.extend(sequences);
    let scan = build_call(m, "scan", &scan_args);
    let traj = bind_fresh(m, scan, "chain_sample");
    let value = field(m, traj, "state");
    let states = field(m, traj, "rng");
    let last = m.alloc(Node::Lit(Scalar::Int(n as i64 - 1)));
    Ok((value, build_call(m, "get0", &[states, last])))
}

fn field(m: &mut Module, record: NodeId, name: &str) -> NodeId {
    let key = m.alloc(Node::Lit(Scalar::Str(name.into())));
    build_call(m, "get", &[record, key])
}

/// Bind `rhs` to a fresh synthetic name, so each reader shares one evaluation.
fn bind_fresh(m: &mut Module, rhs: NodeId, stem: &str) -> NodeId {
    let mut serial = m.node_count();
    let name = loop {
        let name = m.intern(&format!("__{stem}_{serial}"));
        if m.binding_by_name(name).is_none() {
            break name;
        }
        serial += 1;
    };
    m.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public: false,
        synthetic: true,
    });
    m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }))
}
