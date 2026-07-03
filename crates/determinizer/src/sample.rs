//! Sample-side determinisation (spec §07 measure-eval-prims; flatppl-dev
//! flatpdl-determinise.md §6b). `rand(rng, lawof(x))` re-runs `x`'s generative
//! subgraph with each `draw(mᵢ)` replaced by `builtin_sample(rngᵢ, mᵢ, inputᵢ)`,
//! threading one RNG state sequentially in dependency order.
//!
//! This task (single-draw leaf + entry) builds the sampled value as a fresh
//! inline node rather than rewriting the `draw` binding in place — a later
//! task adds binding rewrite for shared-ancestor preservation (a latent used
//! twice sampled once and shared by name). A `record` field's `draw` (inline
//! or reached via a `(%ref self x)` binding reference) is resolved uniformly
//! by [`lower_measure_sample`]'s single `resolve_ref_one` call, mirroring
//! `density::lower_measure_density`'s dispatch.
use crate::density::{
    build_call, build_record, builtin_name, expect_builtin_call, refuse, resolve_ref_one,
    split_kernel_constructor,
};
use crate::refuse::RefuseError;
use flatppl_core::{Module, Node, NodeId, Scalar, Symbol};

/// `rand(rng, lawof(x))` → deterministic sample of x's generative subgraph.
pub(crate) fn lower_rand(m: &mut Module, rand_node: NodeId) -> Result<NodeId, RefuseError> {
    let (rng, measure) = {
        let c = expect_builtin_call(m, rand_node, "rand")
            .ok_or_else(|| refuse(rand_node, m, "expected rand"))?;
        if c.args.len() != 2 {
            return Err(refuse(rand_node, m, "rand expects 2 args (rng, measure)"));
        }
        (c.args[0], c.args[1])
    };
    // Strip lawof: rand samples the LAW of a stochastic subgraph. Refuse lawof of
    // a non-stochastic (Dirac) argument (spec: lawof of a deterministic point).
    let inner = strip_lawof(m, measure)
        .ok_or_else(|| refuse(measure, m, "rand's measure must be lawof(<stochastic>)"))?;
    let (value, _rng_out) = lower_measure_sample(m, inner, rng)?;
    Ok(value)
}

/// `lawof(?m)` → `?m`, resolving one level of `(%ref self x)` indirection first.
fn strip_lawof(m: &Module, node: NodeId) -> Option<NodeId> {
    let (resolved, _) = resolve_ref_one(m, node);
    let c = expect_builtin_call(m, resolved, "lawof")?;
    (c.args.len() == 1).then(|| c.args[0])
}

/// Sample `measure`, threading `rng`; returns `(value_node, advanced_rng_node)`.
fn lower_measure_sample(
    m: &mut Module,
    measure: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    // Resolve a single level of `(%ref self x)` indirection on the measure side,
    // mirroring `density::lower_measure_density`'s dispatch.
    let (resolved, _) = resolve_ref_one(m, measure);
    let op = builtin_name(m, resolved);
    match op {
        Some("record") => lower_record_of_draws_sample(m, resolved, rng),
        Some("draw") => lower_draw(m, resolved, rng),
        // Intractable / deferred set — a later task fills the specific messages
        // (combinators, kchain, etc.), mirroring density's refuse-don't-mislower
        // stance for the sampling side.
        _ => Err(refuse(
            resolved,
            m,
            "sample lowering: unsupported measure construct",
        )),
    }
}

/// `draw(kernel(kwargs))` → `builtin_sample` leaf.
fn lower_draw(
    m: &mut Module,
    draw_node: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let inner_measure = {
        let c = expect_builtin_call(m, draw_node, "draw")
            .ok_or_else(|| refuse(draw_node, m, "expected draw"))?;
        if c.args.len() != 1 {
            return Err(refuse(draw_node, m, "draw expects 1 arg"));
        }
        c.args[0]
    };
    let (ctor, kernel_input) = split_constructor(m, inner_measure).ok_or_else(|| {
        refuse(
            inner_measure,
            m,
            "sample leaf: expected a built-in kernel constructor",
        )
    })?;
    Ok(build_sample_term(m, ctor, kernel_input, rng))
}

/// A primitive constructor call `Normal(mu=…, sigma=…)` → (ctor Const node,
/// record of kwargs). Resolves one level of ref indirection, then delegates
/// the constructor-symbol/kwargs read to `density::split_kernel_constructor`
/// (shared with `build_density_term`'s identical need on the density side).
fn split_constructor(m: &mut Module, measure: NodeId) -> Option<(NodeId, NodeId)> {
    let (resolved, _) = resolve_ref_one(m, measure);
    let (ctor_sym, kwargs) = split_kernel_constructor(m, resolved)?;
    let ctor = m.alloc(Node::Const(ctor_sym));
    let input = build_record(m, &kwargs);
    Some((ctor, input))
}

/// Emit `builtin_sample(rng, ctor, kernel_input)` → `(get0(sample, 0)` =
/// variate, `get0(sample, 1)` = new rng`)`. `builtin_sample` returns a
/// `(variate, new_rngstate)` tuple (spec §07 measure-eval-prims); `get0` is the
/// zero-based container accessor used to project each slot (spec §07
/// "functions": `get0(container, selectors...)`) — there is no separate `get1`
/// primitive in this codebase, so the second slot is `get0(sample, 1)` too,
/// exactly like `density::lower_iid`/`lower_joint` project a positional `cat`
/// slot via `get0(v, i)`.
fn build_sample_term(
    m: &mut Module,
    ctor: NodeId,
    kernel_input: NodeId,
    rng: NodeId,
) -> (NodeId, NodeId) {
    let sample = build_call(m, "builtin_sample", &[rng, ctor, kernel_input]);
    let zero = m.alloc(Node::Lit(Scalar::Int(0)));
    let one = m.alloc(Node::Lit(Scalar::Int(1)));
    let value = build_call(m, "get0", &[sample, zero]);
    let new_rng = build_call(m, "get0", &[sample, one]);
    (value, new_rng)
}

/// `record(f = <draw-ref>, …)`: sample each field's draw in field order,
/// threading the rng, and reassemble the record of sampled values. (Task 1:
/// single field verified end-to-end by the golden test; the loop already
/// generalises to multiple fields — a later task adds shared-ancestor
/// preservation via binding rewrite.)
fn lower_record_of_draws_sample(
    m: &mut Module,
    record_node: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let fields: Vec<(Symbol, NodeId)> = {
        let c = expect_builtin_call(m, record_node, "record")
            .ok_or_else(|| refuse(record_node, m, "expected record"))?;
        c.named.iter().map(|n| (n.name, n.value)).collect()
    };
    let mut cur = rng;
    let mut out_fields = Vec::with_capacity(fields.len());
    for (name, val) in fields {
        // `val` is a `(%ref self <draw-binding>)` or an inline draw;
        // `lower_measure_sample` resolves either uniformly.
        let (v, next) = lower_measure_sample(m, val, cur)?;
        out_fields.push((name, v));
        cur = next;
    }
    Ok((build_record(m, &out_fields), cur))
}
