//! Sample-side determinisation (spec §07 measure-eval-prims; flatppl-dev
//! flatpdl-determinise.md §6b). `rand(rng, lawof(x))` re-runs `x`'s generative
//! subgraph with each `draw(mᵢ)` replaced by `builtin_sample(rngᵢ, mᵢ, inputᵢ)`,
//! threading one RNG state sequentially in dependency order.
//!
//! Independent draws (a `record` of leaf draws each referenced once) are built
//! as fresh inline sample nodes and the orphaned `draw` bindings are swept. A
//! **shared latent** — a `draw`-binding referenced by more than one consumer
//! (two record fields, or another draw's kernel input, i.e. a hierarchical
//! model like `y = draw(Normal(mu = mu, …))`) — must be sampled ONCE and shared
//! by name: [`lower_shared_record_sample`] rewrites each such latent's
//! `draw`-BINDING in place to a single `builtin_sample` (via
//! [`Module::set_binding_rhs`], mirroring `density::lower_record_of_draws`) and
//! lets consumers reference it as `(%ref self mu)`. Inlining a shared latent
//! per consumer would re-draw it and break shared-ancestor identity
//! (measure-algebra-audit H7/M4). A `record` field's `draw` (inline or reached
//! via a `(%ref self x)` binding reference) is resolved uniformly by
//! [`lower_measure_sample`]'s single `resolve_ref_one` call, mirroring
//! `density::lower_measure_density`'s dispatch.
use crate::density::{
    build_call, build_record, builtin_name, draw_argument, expect_builtin_call, refuse,
    resolve_ref_one, split_kernel_constructor,
};
use crate::refuse::RefuseError;
use flatppl_core::{
    Binding, BindingId, Module, NamedKind, Node, NodeId, Ref, RefNs, Scalar, Symbol,
};

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

/// `record(f = <draw-ref>, …)`: sample the record's draws, threading the rng, and
/// reassemble the record of sampled values. If any latent is shared (a
/// `draw`-binding used by two fields or by another draw's kernel input — see
/// [`requires_shared_binding_rewrite`]) this delegates to
/// [`lower_shared_record_sample`], which samples each latent ONCE. Otherwise the
/// independent-draws fold suffices: each field's sample consumes the *previous*
/// field's advanced rng (`cur = next`), not the original `rng` re-read from
/// scratch (verified for >=2 independent draws — Task 2's golden).
///
/// Guards mirror `density::match_independent_record`'s defensive checks
/// (refuse-don't-mislower discipline): a field-keyed measure record has no
/// positional args and only `%field` named entries. The positional-args guard
/// IS reachable via valid surface syntax (`record(a)` inside a `rand`/`lawof`,
/// same as on the density side — see
/// `tests/sample_golden.rs::positional_measure_record_sample_refuses`); the
/// non-`%field` named-arg guard is not (the parser hardcodes `NamedKind::Field`
/// for every named arg inside a `record(...)` call), but is kept so the
/// function stays defensive as later tasks extend it.
fn lower_record_of_draws_sample(
    m: &mut Module,
    record_node: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let fields: Vec<(Symbol, NodeId)> = {
        let c = expect_builtin_call(m, record_node, "record")
            .ok_or_else(|| refuse(record_node, m, "expected record"))?;
        if !c.args.is_empty() {
            return Err(refuse(
                record_node,
                m,
                "record with positional args is not a field-keyed product",
            ));
        }
        let mut fields = Vec::with_capacity(c.named.len());
        for n in c.named.iter() {
            if n.kind != NamedKind::Field {
                return Err(refuse(
                    record_node,
                    m,
                    "non-field named arg in measure record",
                ));
            }
            fields.push((n.name, n.value));
        }
        fields
    };
    // A `draw`-binding referenced by more than one consumer (two fields here, or
    // another draw's kernel input) is a SHARED latent: the per-field inline fold
    // below would sample it once per consumer, re-drawing it and breaking
    // shared-ancestor identity (measure-algebra-audit H7/M4). Detect that and route
    // to the binding-rewrite path, which samples each latent once.
    let field_bids: Vec<Option<BindingId>> = fields
        .iter()
        .map(|&(_, v)| field_draw_binding(m, v))
        .collect();
    if requires_shared_binding_rewrite(m, &field_bids) {
        return lower_shared_record_sample(m, &fields, &field_bids, rng);
    }

    // Independent-draws fold (verified for >=2 independent draws): each field's
    // sample consumes the *previous* field's advanced rng (`cur = next`), not the
    // original `rng` re-read from scratch.
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

/// If `value` is `(%ref self name)` pointing at a binding whose RHS is `draw(…)`,
/// return that binding — the latent this field consumes. Inline-draw and
/// non-draw-ref fields return `None` (they cannot be a *shared* ancestor: an
/// inline draw has a single syntactic site).
fn field_draw_binding(m: &Module, value: NodeId) -> Option<BindingId> {
    if let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = m.node(value)
    {
        let bid = m.binding_by_name(*name)?;
        if draw_argument(m, m.binding(bid).rhs).is_some() {
            return Some(bid);
        }
    }
    None
}

/// Does this record need the shared-latent binding-rewrite path (rather than the
/// independent-draws inline fold)? Yes iff either:
///
/// * a `draw`-binding is referenced by two or more fields (`record(a = mu, b =
///   mu)`), or
/// * a field's draw is *hierarchical* — its kernel input references another
///   `draw`-binding (`y1 = draw(Normal(mu = mu, …))`), which then MUST stay a
///   named binding rather than be inlined.
///
/// Either way the naive fold would re-draw the shared latent (or leave the
/// referenced latent an un-lowered `draw`). Independent leaf draws hit neither.
fn requires_shared_binding_rewrite(m: &Module, field_bids: &[Option<BindingId>]) -> bool {
    let seeds: Vec<BindingId> = field_bids.iter().flatten().copied().collect();
    // A latent referenced by two or more fields.
    for (i, &a) in seeds.iter().enumerate() {
        if seeds[i + 1..].contains(&a) {
            return true;
        }
    }
    // A hierarchical draw whose kernel input references another draw-binding.
    seeds.iter().any(|&bid| {
        draw_argument(m, m.binding(bid).rhs)
            .map(|measure| !referenced_draw_bindings(m, measure).is_empty())
            .unwrap_or(false)
    })
}

/// Sample a record whose fields reference (possibly shared) `draw`-bindings,
/// preserving shared-ancestor identity. Each latent in the generative cone is
/// rewritten to a SINGLE `builtin_sample` bound to a fresh synthetic name; its
/// value (`get0(sample, 0)`) replaces the latent's `draw`-binding RHS and its
/// advanced rng (`get0(sample, 1)`) threads to the next latent. Consumers keep
/// referencing the latent as `(%ref self mu)`, so the shared latent is sampled
/// once and read by name everywhere.
///
/// Binding the full `(value, rng)` sample TUPLE to a name (and projecting both
/// slots by-name-ref) is essential: the FlatPIR writer has no common-subexpression
/// sharing, so an inline sample node shared by NodeId would be textually
/// re-expanded at each `get0` site (re-drawing the latent, and inflating the
/// `builtin_sample` count). This mirrors the parser's `lower_decomposition`, which
/// binds a stochastic source to a shared synthetic name so its slot projections
/// read the *same* draw.
fn lower_shared_record_sample(
    m: &mut Module,
    fields: &[(Symbol, NodeId)],
    field_bids: &[Option<BindingId>],
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    // Latents in dependency (topological) order: a latent is sampled after every
    // draw-binding its kernel input references (spec §07: thread one RNG state
    // sequentially in dependency order).
    let seeds: Vec<BindingId> = field_bids.iter().flatten().copied().collect();
    let cone = topo_draw_cone(m, &seeds);

    let mut cur = rng;
    for &bid in &cone {
        // Read the draw's inner measure BEFORE rewriting the binding (the measure
        // node is a distinct arena node from the `draw` binding RHS, so it survives
        // the rewrite; a later latent's `(%ref self mu)` resolves by name to the
        // now-sampled value).
        let measure = draw_argument(m, m.binding(bid).rhs)
            .ok_or_else(|| refuse(m.binding(bid).rhs, m, "shared-sample: expected a draw"))?;
        let (ctor, kernel_input) = split_constructor(m, measure).ok_or_else(|| {
            refuse(
                measure,
                m,
                "shared-sample latent: expected a built-in kernel constructor",
            )
        })?;

        // sample = builtin_sample(rng_cur, ctor, input), bound to a fresh name so
        // both slots reference it by name (no CSE re-expansion — see fn doc).
        let sample = build_call(m, "builtin_sample", &[cur, ctor, kernel_input]);
        let sample_name = fresh_sample_name(m, bid);
        m.add_binding(Binding {
            name: sample_name,
            rhs: sample,
            doc: None,
            public: false,
            synthetic: true,
        });

        // Rewrite the latent's draw-BINDING to the sampled value; consumers keep
        // their `(%ref self <latent>)` and resolve to it by name.
        let value = get_slot(m, sample_name, 0);
        m.set_binding_rhs(bid, value);

        // Thread the advanced rng from the SAME sample binding into the next latent.
        cur = get_slot(m, sample_name, 1);
    }

    // Assemble the record. A field that references a (now-rewritten) latent keeps
    // its `(%ref self <latent>)` — the shared sample, read by name. Any other field
    // (an inline draw, or a ref to a non-draw binding) is sampled inline, threading
    // the rng after the cone.
    let mut out_fields = Vec::with_capacity(fields.len());
    for (&(name, val), &bid_opt) in fields.iter().zip(field_bids) {
        if bid_opt.is_some() {
            out_fields.push((name, val));
        } else {
            let (v, next) = lower_measure_sample(m, val, cur)?;
            out_fields.push((name, v));
            cur = next;
        }
    }
    Ok((build_record(m, &out_fields), cur))
}

/// `get0((%ref self <name>), slot)` — project slot `slot` of the sample tuple
/// bound to `name`, referencing the binding BY NAME (so the writer does not
/// re-expand the underlying `builtin_sample`). `get0` is the zero-based container
/// accessor; there is no separate `get1` primitive (see [`build_sample_term`]).
fn get_slot(m: &mut Module, name: Symbol, slot: i64) -> NodeId {
    let sample_ref = m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }));
    let idx = m.alloc(Node::Lit(Scalar::Int(slot)));
    build_call(m, "get0", &[sample_ref, idx])
}

/// A fresh private synthetic binding name for a latent's sample tuple, following
/// the parser's `__`-prefixed synthetic convention (`bind_name`) and deduped
/// against existing names.
fn fresh_sample_name(m: &mut Module, latent: BindingId) -> Symbol {
    let latent_name = m.binding(latent).name;
    let base = m.resolve(latent_name).to_string();
    let mut candidate = format!("__sample_{base}");
    let mut n = 1;
    loop {
        let sym = m.intern(&candidate);
        if m.binding_by_name(sym).is_none() {
            return sym;
        }
        candidate = format!("__sample_{base}_{n}");
        n += 1;
    }
}

/// The `draw`-bindings referenced by `(%ref self name)` anywhere in the subtree
/// at `root` (a draw's kernel input), in first-encounter order. Only bindings
/// whose RHS is a `draw(…)` count — a reference to a deterministic binding is not
/// a latent dependency.
fn referenced_draw_bindings(m: &Module, root: NodeId) -> Vec<BindingId> {
    let mut found: Vec<BindingId> = Vec::new();
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            if let Some(bid) = m.binding_by_name(*name) {
                if draw_argument(m, m.binding(bid).rhs).is_some() && !found.contains(&bid) {
                    found.push(bid);
                }
            }
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    found
}

/// The generative cone of draw-bindings reachable from `seeds` (the fields'
/// latents), in dependency (topological) order — each latent appears AFTER every
/// draw-binding its kernel input references, so RNG threading and the
/// sample-once rewrite proceed dependencies-first. Bindings form a DAG (FlatPPL
/// is single-assignment and control-flow-free); a repeated node is emitted once.
fn topo_draw_cone(m: &Module, seeds: &[BindingId]) -> Vec<BindingId> {
    let mut order: Vec<BindingId> = Vec::new();
    let mut visited: Vec<BindingId> = Vec::new();
    for &s in seeds {
        visit_draw_cone(m, s, &mut order, &mut visited);
    }
    order
}

/// Post-order DFS helper for [`topo_draw_cone`]: mark `bid` visited on entry
/// (so a shared latent reached by several dependents is emitted once), recurse
/// into its kernel-input draw dependencies, then push `bid`.
fn visit_draw_cone(
    m: &Module,
    bid: BindingId,
    order: &mut Vec<BindingId>,
    visited: &mut Vec<BindingId>,
) {
    if visited.contains(&bid) {
        return;
    }
    visited.push(bid);
    if let Some(measure) = draw_argument(m, m.binding(bid).rhs) {
        for dep in referenced_draw_bindings(m, measure) {
            visit_draw_cone(m, dep, order, visited);
        }
        order.push(bid);
    }
}
