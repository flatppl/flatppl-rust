//! Sample-side determinisation (spec §07 measure-eval-prims). `rand(lawof(x), rng)`
//! re-runs `x`'s generative subgraph with each `draw(mᵢ)` replaced by
//! `builtin_sample(rngᵢ, mᵢ, inputᵢ)`,
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
//! (measure-algebra audit, 2026-06, findings H7/M4). A `record` field's `draw` (inline or reached
//! via a `(%ref self x)` binding reference) is resolved uniformly by
//! [`lower_measure_sample`]'s single `resolve_ref_one` call, mirroring
//! `density::lower_measure_density`'s dispatch.
//!
//! **Two dispatchers, two admissible sets.** [`lower_closed_measure_sample`]
//! handles MEASURE position — `rand`'s first argument, which §07 types as "a
//! closed measure `m`", so a bare `Normal(…)` or a `pushfwd(f, M)` is admissible
//! there, not just a `lawof`. [`lower_measure_sample`] handles FIELD position,
//! where §04 says a record field is a declared variate or a deterministic
//! expression — so an un-drawn measure in a field keeps refusing rather than
//! fabricating a draw. A derived field (`d = y1 - y2`) passes through unchanged
//! (§13 "Output reduction") with the latents it reads sampled once each and
//! referenced by name.
use crate::density::{
    build_call, build_record, build_user_call, builtin_name, draw_argument, expect_builtin_call,
    iid_static_size, refuse, resolve_ref_one, split_kernel_constructor,
};
use crate::refuse::RefuseError;
use flatppl_core::{
    Binding, BindingId, CallHead, Module, NamedKind, Node, NodeId, Phase, Ref, RefNs, Scalar,
    Symbol, Type,
};

/// `rand(m, rng)` → a deterministic sample of the closed measure `m`.
///
/// The measure is handed straight to [`lower_closed_measure_sample`], which
/// dispatches on its head — `lawof(x)` (re-run `x`'s generative subgraph),
/// `pushfwd(f, M)`, a `record`/`draw`, or a bare primitive constructor. §07 types
/// this argument as "a closed measure `m`", so `lawof` is one admissible spelling
/// among several rather than a requirement; gating on a literal `lawof` head
/// before dispatching is what used to make every other closed measure
/// unreachable.
///
/// `bid` is the binding whose subtree contains `rand_node` (the driver's
/// `apply_rule` already has it) — i.e. the name a `v, s2 = rand(...)`
/// decomposition or a bare `draws = rand(...)` assignment binds the `rand`
/// call to. It names a restored law and selects the result shape through
/// [`rand_result_is_destructured`] (tuple vs bare value).
pub(crate) fn lower_rand(
    m: &mut Module,
    bid: BindingId,
    rand_node: NodeId,
    preserve_tuple: bool,
) -> Result<NodeId, RefuseError> {
    let (measure, rng) = {
        let c = expect_builtin_call(m, rand_node, "rand")
            .ok_or_else(|| refuse(rand_node, m, "expected rand"))?
            .clone();
        let names = [m.intern("m"), m.intern("rstate")];
        let args = crate::kernel::bind_call_args(m, &names, &c.args, &c.named, false)
            .ok_or_else(|| refuse(rand_node, m, "rand expects 2 args (measure, rng)"))?;
        (args[0], args[1])
    };
    let measure = instantiate_declared_draws(m, measure, bid);
    let (value, rng_out) = lower_closed_measure_sample(m, measure, rng)?;
    let result = if preserve_tuple || rand_result_is_destructured(m, bid) {
        // Full spec §07 (value, new_rstate) contract: retain both slots for
        // host outputs or the parser's get(_,1)/get(_,2) projections.
        build_call(m, "tuple", &[value, rng_out])
    } else {
        // Value-terminal shortcut: `draws = rand(...)` used as a bare value /
        // read by string selector — return the bare value (unchanged).
        value
    };
    m.mark_sample_result(result);
    Ok(result)
}

/// A previous query's pins are realizations, not this sample's generative law.
/// Copy an affected draw cone under private names so lowering this query cannot
/// replace the values read by an earlier sampled output. Unpinned graphs keep
/// their existing bindings and lowering path.
fn instantiate_declared_draws(m: &mut Module, measure: NodeId, query: BindingId) -> NodeId {
    let mut pending = vec![measure];
    let mut seen = std::collections::HashSet::new();
    let mut seen_bindings = std::collections::HashSet::new();
    let mut draws = Vec::new();
    let mut pinned = false;
    while let Some(node) = pending.pop() {
        if !seen.insert(node) || is_rand_boundary(m, node) {
            continue;
        }
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(node)
            && let Some(binding) = m.binding_by_name(*name)
            && seen_bindings.insert(binding)
        {
            pinned |= m.query_pinned_rhs(binding).is_some();
            let rhs = m.declared_binding_rhs(binding);
            if draw_argument(m, rhs).is_some() {
                draws.push((binding, rhs));
            }
            pending.push(rhs);
        }
        m.for_each_child(node, |child| pending.push(child));
    }
    if !pinned || draws.is_empty() {
        return measure;
    }

    let mut replacements = Vec::with_capacity(draws.len());
    let mut copies = Vec::with_capacity(draws.len());
    for (binding, rhs) in draws {
        let original = Ref {
            ns: RefNs::SelfMod,
            name: m.binding(binding).name,
        };
        let name = fresh_latent_name(m, binding, "draw");
        copies.push(m.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: false,
            synthetic: true,
        }));
        let value = m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }));
        replacements.push((original, value));
    }
    for binding in copies {
        let rhs = crate::kernel::substitute_admitted(
            m,
            m.binding(binding).rhs,
            &replacements,
            crate::kernel::Substitute::DeclaredDraws,
            true,
        );
        m.set_binding_rhs(binding, rhs);
    }
    let measure = crate::kernel::substitute_admitted(
        m,
        measure,
        &replacements,
        crate::kernel::Substitute::DeclaredDraws,
        true,
    );
    let name = fresh_latent_name(m, query, "law");
    m.add_binding(Binding {
        name,
        rhs: measure,
        doc: None,
        public: false,
        synthetic: true,
    });
    // Recompute phase and static iid shapes on the restored graph. The temporary
    // law binding makes its copied expression reachable to inference and is swept
    // once this query lowers.
    let _ = flatppl_infer::infer(m);
    measure
}

/// An explicit query is a concrete value, before or after its own lowering.
pub(crate) fn is_rand_boundary(m: &Module, node: NodeId) -> bool {
    m.is_sample_result(node) || expect_builtin_call(m, node, "rand").is_some()
}

/// Is `rand_bid`'s value DESTRUCTURED — read via an INTEGER-literal tuple
/// projection (`get(_, k)` / `get0(_, k)`) rather than used as a bare value?
///
/// `rand(lawof(x), rng)` infers to `Tuple([domain(x), RngState])` (spec §07;
/// `crates/infer/src/ops.rs`'s `"rand"` phase arm). [`lower_rand`] uses this
/// predicate to DISPATCH the result shape: true builds the full 2-tuple
/// `tuple(value, advanced_rng)`; false returns the bare sampled value, dropping
/// the advanced rng. The parser's `v, s2 = rand(...)` decomposition sugar
/// (`lower_decomposition`, `crates/syntax/src/parser.rs`) lowers to exactly
/// `__0x1 = rand(...); v = get(__0x1, 1); s2 = get(__0x1, 2)` — a synthetic
/// tmp binding (name pattern `__0x<hex>`) plus 1-based integer-literal `get`
/// projections off it. A user can write the same shape directly with the
/// 0-based `get0(draws, 0)` / `get0(draws, 1)`. Getting this dispatch wrong in
/// the value-terminal direction would erase the tuple and substitute the bare
/// value in `rand_bid`'s place, leaving a surviving `get(<rand-value>, 1)` (or
/// `get0(<rand-value>, 0)`, etc.) indexing a NON-tuple — wrong/out-of-range
/// FlatPDL emitted SILENTLY, since the determiniser does not re-infer after the
/// rewrite and `is_flatpdl` is structural (whole-branch review finding:
/// "silent mislowering"). This predicate is what keeps the two paths sound.
///
/// A STRING-literal selector (`get(draws, "mu")` / `draws.mu`, record-field
/// access) is a DIFFERENT selector shape — `get_type`'s `Type::Record` arm
/// keys on `Node::Lit(Scalar::Str(_))`, never `Scalar::Int` — so it is not a
/// tuple projection and must NOT trip this predicate: the value-terminal
/// convention (`draws` standing in for the record `lower_rand` returns) still
/// needs its fields readable by name.
fn rand_result_is_destructured(m: &Module, rand_bid: BindingId) -> bool {
    let name = m.binding(rand_bid).name;
    m.bindings()
        .any(|(_, binding)| subtree_has_int_projection_of(m, binding.rhs, name))
}

/// BFS the subtree at `root` for a `get`/`get0` call whose first argument is
/// `(%ref self name)` and whose selector argument is an integer literal — see
/// [`rand_result_is_destructured`]. Mirrors the worklist-over-`for_each_child`
/// idiom used throughout this crate (e.g. this file's own
/// [`referenced_draw_bindings`], `Module::collect_referenced_names`).
fn subtree_has_int_projection_of(m: &Module, root: NodeId, name: Symbol) -> bool {
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        if is_int_projection_of(m, id, name) {
            return true;
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    false
}

/// True iff `id` is `get(subject, k)` or `get0(subject, k)` where `subject` is
/// `(%ref self name)` and `k` is an integer literal — a tuple-slot projection
/// of the binding named `name`, as opposed to a string-selector field access.
fn is_int_projection_of(m: &Module, id: NodeId, name: Symbol) -> bool {
    let Some(c) = expect_builtin_call(m, id, "get").or_else(|| expect_builtin_call(m, id, "get0"))
    else {
        return false;
    };
    if c.args.len() < 2 {
        return false;
    }
    let subject_is_name = matches!(
        m.node(c.args[0]),
        Node::Ref(Ref { ns: RefNs::SelfMod, name: n }) if *n == name
    );
    subject_is_name && matches!(m.node(c.args[1]), Node::Lit(Scalar::Int(_)))
}

/// `lawof(?m)` → `?m`, resolving one level of `(%ref self x)` indirection first.
fn strip_lawof(m: &Module, node: NodeId) -> Option<NodeId> {
    let (resolved, _) = resolve_ref_one(m, node);
    let c = expect_builtin_call(m, resolved, "lawof")?;
    (c.args.len() == 1).then(|| c.args[0])
}

/// Sample a MEASURE-POSITION measure — `rand`'s first argument, or the base of
/// a `pushfwd` — threading `rng`; returns `(value_node, advanced_rng_node)`.
///
/// §07 `rand(m, rstate)` types this argument as "a closed measure `m`", and §06
/// ("Fundamental measures and measure algebra") makes a primitive constructor
/// call a nullary kernel, i.e. closed; measure-to-measure operations preserve
/// arity, so `pushfwd(f, <closed>)` is closed too. §07's only stated exclusions
/// are "measures involving non-constant weighting … or multivariate truncation",
/// both still refused via [`classify_intractable_or_deferred`].
///
/// Separate from [`lower_measure_sample`], which doubles as the per-FIELD dispatcher
/// for a record law, because the two positions admit different sets (§04's variate /
/// measure distinction). Putting the primitive-constructor leaf in the shared
/// dispatcher would make `lawof(record(a = Normal(…)))` sample a field the model never
/// declared as a variate
/// (`tests/refuse.rs::undrawn_measure_in_a_record_field_still_refuses`).
fn lower_closed_measure_sample(
    m: &mut Module,
    measure: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let (resolved, _) = resolve_ref_one(m, measure);
    match builtin_name(m, resolved) {
        Some("lawof") => {
            let inner = strip_lawof(m, resolved)
                .ok_or_else(|| refuse(resolved, m, "lawof expects 1 arg"))?;
            // Test `?x`'s phase, not `lawof(?x)`'s: §04 "Phase of the reified law"
            // makes `lawof` absorb its argument's stochasticity, so `lawof(?x)` is
            // always deterministic-phase. A non-Stochastic `?x` (`lawof(3.0)`) has no
            // generative `draw` subgraph for `rand` to re-run.
            //
            // Belongs to the `lawof` ARM, not to `rand` as a whole: a bare `Normal(…)`
            // in measure position is deterministic-phase too, so a rand-level phase
            // check would refuse every composed measure.
            if !matches!(m.phase_of(inner), Some(Phase::Stochastic)) {
                return Err(refuse(
                    inner,
                    m,
                    "lawof's argument is not stochastic-phase (a Dirac/deterministic point), so it \
                     has no generative draw for `rand` to sample",
                ));
            }
            lower_closed_measure_sample(m, inner, rng)
        }
        Some("pushfwd") => lower_pushfwd_sample(m, resolved, rng),
        Some("rebase") => lower_rebase_sample(m, resolved, rng, None),
        // §07's own worked `rand` example is `rand(iid(Normal(0, 1), 10), rstate)`:
        // a fixed-kernel `iid` is a nullary kernel, hence closed. Routes to the
        // SAME [`lower_iid_sample`] as the `draw(iid(K, n))` spelling, so both emit
        // the identical batched term.
        Some("iid") => {
            let (kernel, iid_node) = split_iid(m, resolved)
                .ok_or_else(|| refuse(resolved, m, "iid expects 2 args (kernel, count)"))?;
            lower_iid_sample(m, kernel, iid_node, rng, None)
        }
        Some("record") => lower_record_of_draws_sample(m, resolved, rng),
        Some("draw") => lower_draw(m, resolved, rng, None),
        _ => {
            // The intractable/deferred set first — `weighted(w, M)` and friends
            // carry positional args, so `split_constructor` would reject them
            // anyway, but with the generic "not a constructor" message instead of
            // the actionable §07 one.
            if let Some(err) = classify_intractable_or_deferred(m, resolved) {
                return Err(err);
            }
            // The leaf: a primitive constructor call IS a closed measure (§06), so
            // `rand(Normal(mu = 0.0, sigma = 1.0), s)` samples it directly with no
            // `lawof` wrapper. This is also what terminates [`lower_pushfwd_sample`]'s
            // recursion on its base.
            if let Some((ctor, kernel_input)) = split_constructor(m, resolved) {
                return Ok(build_sample_term(m, ctor, kernel_input, rng, None));
            }
            Err(refuse(
                resolved,
                m,
                "sample lowering: unsupported measure construct",
            ))
        }
    }
}

/// `pushfwd(f, M)` in measure position → sample `M`, then apply `f` FORWARD to the
/// sampled value; the rng is the one `M`'s sample advanced.
///
/// $X \sim M \Rightarrow f(X) \sim f_* M$ by the definition of the pushforward (§06
/// `pushfwd`: $(f_* M)(Y) = M(f^{-1}(Y))$), so sampling needs **no inverse and no
/// Jacobian** — this path must never touch `crate::invert` or the bijection registry
/// the density side uses, and `f` need not be injective. Nested `pushfwd`s recurse to
/// [`lower_closed_measure_sample`]'s primitive-constructor leaf.
///
/// The forward application is `(%call f <sampled value>)`, and only two forms resolve
/// to FlatPDL:
///
/// * a **bare builtin** callee (`pushfwd(exp, …)`) — `canon::inline`'s
///   `builtin_callee_head` rewrites the head to a direct builtin call;
/// * a **reified** `functionof`/lambda that beta-reduces under
///   [`crate::kernel::reduce_kernel_application`], applied here rather than left to the
///   driver's fixpoint so a map that does NOT reduce refuses HERE, naming the map.
///
/// A surviving `CallHead::User` refuses at exit anyway
/// (`NonConformKind::ResidualUserCall`); refusing here gives the specific reason.
fn lower_pushfwd_sample(
    m: &mut Module,
    pushfwd_node: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let (map, base) = {
        let c = expect_builtin_call(m, pushfwd_node, "pushfwd")
            .ok_or_else(|| refuse(pushfwd_node, m, "expected pushfwd"))?;
        if c.args.len() != 2 {
            return Err(refuse(
                pushfwd_node,
                m,
                "pushfwd expects 2 args (map, base measure)",
            ));
        }
        // `pushfwd(f, M)` is function-FIRST (§06): the map is arg 0, the base
        // measure arg 1.
        (c.args[0], c.args[1])
    };
    let (value, rng_out) = lower_closed_measure_sample(m, base, rng)?;
    // A bare builtin map needs no reduction — the head rewrite handles it — EXCEPT over
    // a record variate, where the application is §04 auto-splatting against that
    // operator's own argument names. `flatppl_infer` exposes ordered parameter names
    // only for measure constructors; a deterministic builtin's catalogue row is
    // `Sig::Function { params: Vec<ParamSig> }` — argument TYPES, no names.
    //
    // So the whole record-valued case refuses, over-refusing a name-matching map like
    // `pushfwd(exp, lawof(record(x = y1)))`. Nothing downstream would catch a MISmatch:
    // `infer` reports no diagnostic and `is_flatpdl` is structural, so
    // `exp(record(y = …))` passes every gate and fails only in the engine.
    if matches!(m.node(map), Node::Const(_)) {
        if expect_builtin_call(m, value, "record").is_some() {
            return Err(refuse(
                pushfwd_node,
                m,
                "pushfwd's map is a bare built-in over a record variate: §04 auto-splatting needs \
                 argument names the catalogue lacks — use a functionof or lambda over the fields",
            ));
        }
        return Ok((build_user_call(m, map, value), rng_out));
    }
    // For a RECORD-valued variate the forward application is an auto-splatting
    // call, whose §04 correspondence rule the reducer does not enforce — check it
    // before reducing (see [`record_splat_mismatch`]).
    if let Some(kernel) = crate::kernel::resolve_reified(m, map)
        && let Some(why) = record_splat_mismatch(m, value, &kernel.inputs)
    {
        return Err(refuse(
            pushfwd_node,
            m,
            &format!(
                "pushfwd's map does not correspond to its base measure's record variate \
                     ({why}); §04 \"Calling conventions\" makes a field/argument-name mismatch a \
                     static error"
            ),
        ));
    }
    let applied = build_user_call(m, map, value);
    let reduced = crate::kernel::reduce_kernel_application(m, applied).ok_or_else(|| {
        refuse(
            pushfwd_node,
            m,
            "pushfwd's map does not reduce at the sampled variate: its parameters do not bind the \
             base domain (§04 binds a record's fields by name), leaving a residual call",
        )
    })?;
    Ok((reduced, rng_out))
}

/// Do the map's parameter names correspond EXACTLY to the fields of a
/// record-valued variate? Returns `None` when they do, or when `value` is not a
/// record literal (there is nothing to splat); otherwise the offending name, for
/// the refusal message.
///
/// §04 "Calling conventions": "A call with field or column names that do not match the
/// callable's argument names is a static error". [`crate::kernel::reduce_kernel_application`]'s
/// record branch binds each input from a like-named field and never checks that every
/// FIELD was consumed, so a map declaring a strict SUBSET reduces to a projection of
/// the variate — a different measure, hence a wrong sample.
fn record_splat_mismatch(m: &Module, value: NodeId, inputs: &[(Symbol, Ref)]) -> Option<String> {
    let fields: Vec<Symbol> = {
        let c = expect_builtin_call(m, value, "record")?;
        c.named.iter().map(|n| n.name).collect()
    };
    let params: Vec<Symbol> = inputs.iter().map(|&(n, _)| n).collect();
    // Name BOTH sides in the message. Blaming the odd name alone reads absurdly for
    // a map whose whole purpose is selecting that coordinate ("variate field `y1`
    // binds no map parameter", for a map that projects onto `y1`); showing the two
    // lists makes it a correspondence failure, which is what it is.
    let listing = |syms: &[Symbol]| {
        syms.iter()
            .map(|&s| format!("`{}`", m.resolve(s)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let sides = format!(
        "map parameters {} vs variate fields {}",
        listing(&params),
        listing(&fields)
    );
    if let Some(f) = fields.iter().find(|f| !params.contains(f)) {
        return Some(format!(
            "{sides} — field `{}` binds no parameter",
            m.resolve(*f)
        ));
    }
    if let Some(p) = params.iter().find(|p| !fields.contains(p)) {
        return Some(format!(
            "{sides} — parameter `{}` names no field",
            m.resolve(*p)
        ));
    }
    None
}

/// Sample `measure`, threading `rng`; returns `(value_node, advanced_rng_node)`.
///
/// This is the per-FIELD dispatcher for a record law — see
/// [`lower_closed_measure_sample`] for why measure position has its own entry
/// point and why the primitive-constructor leaf must not be added here.
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
        Some("draw") => lower_draw(m, resolved, rng, None),
        // Intractable (§07's `rand` exclusion clause + §13 "Refused constructs") /
        // deferred (simply not built in this vertical) — see
        // `classify_intractable_or_deferred`.
        // This dispatch arm is reached when one of these ops is `lawof`'s direct
        // argument, or a NOT-yet-drawn measure sitting in a record field (the
        // uniform per-field fold in `lower_record_of_draws_sample`/
        // `lower_shared_record_sample` calls back into this function for every
        // field, regardless of what op the field's value resolves to). The far
        // more common surface shape — `draw(weighted(...))`,
        // `draw(truncate(...))`, etc. — is classified the SAME way from inside
        // `lower_draw`, since that path never reaches this `match` at all (see
        // that function's doc comment).
        _ => Err(
            classify_intractable_or_deferred(m, resolved).unwrap_or_else(|| {
                refuse(
                    resolved,
                    m,
                    "sample lowering: unsupported measure construct",
                )
            }),
        ),
    }
}

/// Classify a RESOLVED measure node as one of the ops this sample vertical
/// intentionally does not lower, or `None` if `resolved`'s builtin head is not
/// one of them (the caller then falls back to its own generic refuse).
///
/// Two buckets. There is no enumerated "tractable set" in the spec to check
/// against — §07 "rand" states a CRITERION: "`rand` does not support measures
/// for which [efficient IID generation] is an intractable problem, especially
/// measures involving non-constant weighting (via `weighted(f, base)`,
/// `logweighted(g, base)`, or `bayesupdate(L, prior)`) or multivariate
/// truncation" — "especially" introduces examples, not a whitelist. §13
/// "Refused constructs" separately and explicitly names the same two shapes as
/// refused for a sampled output. Only those two are classified **intractable**
/// here; everything else this function refuses is **deferred** (not
/// spec-excluded, simply unbuilt):
/// * **Intractable** (§07 "rand"'s exclusion clause; §13 "Refused
///   constructs") — `weighted`/`logweighted`/`bayesupdate` (a reweighted
///   measure has no direct sampling recipe; realizing its law needs a
///   change-of-measure algorithm — rejection/importance sampling, MCMC — which
///   is out of scope for this MVP's exact, deterministic lowering), and a
///   `truncate` whose base is CONFIRMED multivariate (no general sampling
///   recipe for an arbitrary multivariate truncated region either).
/// * **Deferred** — `jointchain`/`kchain`/`superpose`, and a univariate
///   `truncate`: none of these are spec-excluded (a later vertical could add
///   inverse-CDF/rejection truncated sampling, or thread the rng through a
///   Kleisli/joint chain), they are simply not built in this one
///   (direct draws + record-of-draws + shared ancestors + pushforwards).
///
/// `pushfwd` is deliberately NOT in either bucket: in measure position it is
/// sampled ([`lower_pushfwd_sample`]), and in `draw` position it is refused for a
/// different, semantic reason ([`refuse_draw_of_pushfwd`]).
///
/// Shared by [`lower_measure_sample`]'s dispatcher,
/// [`lower_closed_measure_sample`]'s fallback, and (via
/// [`classify_drawn_measure`]) the two places a `draw`'s inner measure is read.
fn classify_intractable_or_deferred(m: &Module, resolved: NodeId) -> Option<RefuseError> {
    if let Some(lift) = applied_ksuperpose_under_normalize(m, resolved) {
        return Some(refuse_ksuperpose_sample(m, lift));
    }
    match builtin_name(m, resolved) {
        Some("weighted") | Some("logweighted") | Some("bayesupdate") => {
            Some(refuse_weighted_family(m, resolved))
        }
        Some("truncate") => Some(refuse_truncate(m, resolved)),
        Some("jointchain") | Some("kchain") | Some("superpose") => {
            Some(refuse_deferred_combinator(m, resolved))
        }
        _ => None,
    }
}

/// Classify the inner measure of a `draw` — [`classify_intractable_or_deferred`]
/// plus the `pushfwd` case, which is refused in DRAW position (a fresh draw whose
/// base law shares latents with the surrounding model) even though it is sampled
/// in MEASURE position. Shared by [`lower_draw`] and [`lower_shared_record_sample`]'s
/// latent loop — the two places a `draw`'s measure is read.
fn classify_drawn_measure(m: &Module, resolved: NodeId) -> Option<RefuseError> {
    if matches!(builtin_name(m, resolved), Some("pushfwd")) {
        return Some(refuse_draw_of_pushfwd(m, resolved));
    }
    classify_intractable_or_deferred(m, resolved)
}

/// `draw(pushfwd(f, M))`: refused — and NOT because a pushforward is hard to
/// sample (in measure position it is the easy case; see
/// [`lower_pushfwd_sample`]).
///
/// Per §04 a `draw` introduces a **fresh** stochastic node, so
/// `d = draw(pushfwd(f, lawof(record(y1 = y1, y2 = y2))))` is independent of the
/// existing `y1`/`y2`, not a spelling of `d = f(record(y1 = y1, y2 = y2))`. Reusing
/// those samples would emit a perfectly dependent joint where the model asked for
/// independence. Drawing it needs the base law sampled a SECOND time, independently of
/// the surrounding model's copies of the same latents, which this vertical does not
/// build.
fn refuse_draw_of_pushfwd(m: &Module, id: NodeId) -> RefuseError {
    refuse(
        id,
        m,
        "draw(pushfwd(f, M)) is a fresh, independent draw from the pushforward law, not a function \
         of the draws inside M; resampling M independently is not implemented",
    )
}

/// The applied `ksuperpose(K, w)(θ)` node inside `resolved`, looking through one
/// `normalize` wrapper — `normalize(ksuperpose(…)(…))` is §06's own spelling of
/// a mixture, so it is the shape a sample query actually carries. Returns the
/// LIFT node, so the refusal points at the construct that is not sampled rather
/// than at the enclosing `normalize`.
fn applied_ksuperpose_under_normalize(m: &Module, resolved: NodeId) -> Option<NodeId> {
    let mut node = resolved;
    if let Some(c) = expect_builtin_call(m, node, "normalize") {
        if c.args.len() != 1 {
            return None;
        }
        node = resolve_ref_one(m, c.args[0]).0;
    }
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    let CallHead::User(callee) = c.head else {
        return None;
    };
    let (lift_node, _) = resolve_ref_one(m, callee);
    expect_builtin_call(m, lift_node, "ksuperpose").map(|_| lift_node)
}

/// `ksuperpose`: refused in SAMPLE mode, and not because it is intractable.
///
/// §06 is explicit the other way: "Because the weights do not depend on the
/// variate, the mixture is sampleable whenever `kernel` is." Drawing it is the
/// two-stage ancestral sample — pick a component index with probability
/// proportional to `w`, then sample that component — which needs a discrete
/// index sampler threaded through the family axis, and that is not built. So this
/// refuses as UNIMPLEMENTED, deliberately worded apart from
/// [`refuse_weighted_family`]'s intractability: a reader must not conclude the
/// mixture cannot be sampled in principle.
fn refuse_ksuperpose_sample(m: &Module, id: NodeId) -> RefuseError {
    refuse(
        id,
        m,
        "sampling a ksuperpose mixture is not implemented (§06 makes it sampleable \
         whenever the component kernel is, via a component-index draw, but the \
         two-stage sampler over the family axis is not built)",
    )
}

/// `weighted`/`logweighted`/`bayesupdate`: refused per §07 "rand"'s exclusion
/// clause and §13 "Refused constructs" — see [`classify_intractable_or_deferred`].
fn refuse_weighted_family(m: &Module, id: NodeId) -> RefuseError {
    refuse(
        id,
        m,
        "sampling a weighted/logweighted/bayesupdate measure is intractable (§07 \"rand\" names \
         non-constant weighting; §13 \"Refused constructs\" names it explicitly)",
    )
}

/// `truncate(base, S)`: dispatch on whether `base`'s domain is CONFIRMED
/// multivariate (intractable) or not (deferred — see
/// [`classify_intractable_or_deferred`]).
fn refuse_truncate(m: &Module, id: NodeId) -> RefuseError {
    if truncate_is_confirmed_multivariate(m, id) {
        refuse(
            id,
            m,
            "sampling a multivariate truncated measure is intractable (§07 \"rand\" names \
             multivariate truncation; §13 \"Refused constructs\" names it explicitly)",
        )
    } else {
        refuse_deferred_combinator(m, id)
    }
}

/// Is `truncate_node`'s inferred domain CONFIRMED multivariate (`Array`/
/// `TVector`, e.g. a truncated `MvNormal`)? `truncate(base, S)` infers to
/// `fresh_measure(arg_ty(base))` (`crates/infer/src/ops.rs`), so the truncate
/// node's OWN domain already reflects `base`'s — no need to resolve `base`
/// separately. `false` for a confirmed-scalar OR an unresolved/deferred domain:
/// unlike the density-side cat-slice hazard this deliberately does NOT mirror
/// (`lower_joint`'s fail-closed "unconfirmed ⇒ refuse too"), there is no silent
/// mislowering risk here in either direction — an unresolved-domain `truncate`
/// simply falls through to the (still-a-refusal) deferred-combinator message
/// instead of the intractable one.
fn truncate_is_confirmed_multivariate(m: &Module, truncate_node: NodeId) -> bool {
    matches!(
        m.type_of(truncate_node),
        Some(Type::Measure { domain, .. })
            if matches!(domain.as_ref(), Type::Array { .. } | Type::TVector { .. })
    )
}

/// `jointchain`/`kchain`/`superpose`, and a univariate `truncate` (via
/// [`refuse_truncate`]): sample lowering for these is simply not built in this
/// vertical — see [`classify_intractable_or_deferred`].
fn refuse_deferred_combinator(m: &Module, id: NodeId) -> RefuseError {
    refuse(
        id,
        m,
        "sample lowering for this combinator is deferred to the full sample path (this \
         vertical covers direct draws + record-of-draws + shared ancestors)",
    )
}

/// `draw(kernel(kwargs))` → `builtin_sample` leaf.
///
/// **`draw(<op>(...))` is the common surface shape for an intractable/deferred
/// measure** — `draw(weighted(w, M))`, `draw(truncate(M, S))`,
/// `draw(superpose(...))`, etc. — far more common than one of these ops
/// appearing un-drawn (`lower_measure_sample`'s own dispatch arm, reached only
/// when the op is `lawof`'s direct argument or an un-drawn record field). This
/// function's inner measure is read straight off `draw`'s argument, never
/// routed back through [`lower_measure_sample`]'s dispatcher, so WITHOUT the
/// classification below, `draw(weighted(...))` would instead fall through to
/// [`split_constructor`]'s generic "expected a built-in kernel constructor"
/// message: true (a `weighted(...)` call is not a leaf constructor), but not
/// the ACTIONABLE reason. Classify the resolved inner measure the same way
/// [`lower_measure_sample`] does first, so this shape gets the same message.
/// A shared record supplies `sample_name` to bind the primitive tuple before
/// either slot is projected or the sampled value is transformed.
fn lower_draw(
    m: &mut Module,
    draw_node: NodeId,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> Result<(NodeId, NodeId), RefuseError> {
    let inner_measure = {
        let c = expect_builtin_call(m, draw_node, "draw")
            .ok_or_else(|| refuse(draw_node, m, "expected draw"))?;
        if c.args.len() != 1 {
            return Err(refuse(draw_node, m, "draw expects 1 arg"));
        }
        c.args[0]
    };
    let (inner_resolved, _) = resolve_ref_one(m, inner_measure);
    if let Some(err) = classify_drawn_measure(m, inner_resolved) {
        return Err(err);
    }
    // Fan-out: `draw(iid(K, n))` → ONE batched `builtin_sample`, via the same
    // [`lower_iid_sample`] the measure-position `rand(iid(K, n), s)` spelling uses.
    if let Some((kernel, iid_node)) = split_iid(m, inner_measure) {
        return lower_iid_sample(m, kernel, iid_node, rng, sample_name);
    }
    if builtin_name(m, inner_resolved) == Some("rebase") {
        return lower_rebase_sample(m, inner_resolved, rng, sample_name);
    }
    let (ctor, kernel_input) = split_constructor(m, inner_measure).ok_or_else(|| {
        refuse(
            inner_measure,
            m,
            "sample leaf: expected a built-in kernel constructor",
        )
    })?;
    Ok(build_sample_term(m, ctor, kernel_input, rng, sample_name))
}

/// Centering iid normals gives covariance sigma²(I - 11'/n); sorting them
/// gives the normalized ordered chamber. Both consume one batched RNG draw.
fn lower_rebase_sample(
    m: &mut Module,
    measure: NodeId,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> Result<(NodeId, NodeId), RefuseError> {
    use crate::density::rebase::{NormalSet, elementwise, normal_on_set};

    let normal = normal_on_set(m, measure).ok_or_else(|| {
        refuse(
            measure,
            m,
            "rebase sampling currently requires fixed-size iid Normal on Lebesgue(ordered(n)) or Lebesgue(zerosum(n))",
        )
    })?;
    let mu = match normal.kind {
        NormalSet::Ordered => normal.mu,
        NormalSet::ZeroSum => m.alloc(Node::Lit(Scalar::Real(0.0))),
    };
    let sigma = match normal.kind {
        NormalSet::Ordered => normal.sigma,
        NormalSet::ZeroSum => m.alloc(Node::Lit(Scalar::Real(1.0))),
    };
    let parameters = [(m.intern("mu"), mu), (m.intern("sigma"), sigma)];
    let input = build_record(m, &parameters);
    let ctor = m.intern("Normal");
    let ctor = m.alloc(Node::Const(ctor));
    let (value, rng_out) = build_iid_sample_term(m, ctor, input, normal.n, rng, sample_name);
    let value = match normal.kind {
        NormalSet::Ordered => build_call(m, flatppl_core::SORT_VECTOR, &[value]),
        NormalSet::ZeroSum => {
            let sum = build_call(m, "sum", &[value]);
            let n = m.alloc(Node::Lit(Scalar::Real(normal.n as f64)));
            let mean = build_call(m, "divide", &[sum, n]);
            let centered = elementwise(m, "sub", &[value, mean]);
            elementwise(m, "mul", &[centered, normal.sigma])
        }
    };
    Ok((value, rng_out))
}

/// `iid(K, n)` with a FIXED kernel `K` and a static length `n` → ONE batched
/// `builtin_sample(rng, ctor, input, n)` (spec §07 measure-eval-prims:
/// `builtin_sample`'s size-dims form returns an IID array `X` of size `n` with a
/// SINGLE advanced `new_rngstate`, not one per element).
///
/// Shared by BOTH spellings of the same measure, so they emit the identical term:
/// [`lower_draw`]'s `draw(iid(K, n))` and [`lower_closed_measure_sample`]'s
/// measure-position `rand(iid(K, n), s)` — §07's own worked `rand` example
/// (`random_data, rstate2 = rand(iid(Normal(0, 1), 10), rstate)`), admissible
/// because a fixed-kernel `iid` is itself a nullary kernel, i.e. a closed measure.
///
/// [`split_constructor`] is what rejects a kernel that is not a bare built-in
/// constructor call — in particular a `broadcast(K, arr0, arr1, …)` kernel (an
/// array-of-kernels measure with DIFFERING per-element params, §04 broadcasting)
/// has positional args, so it is refused rather than mislowered as a fixed-kernel
/// fan-out.
fn lower_iid_sample(
    m: &mut Module,
    kernel: NodeId,
    iid_node: NodeId,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> Result<(NodeId, NodeId), RefuseError> {
    let n = iid_static_size(m, iid_node).ok_or_else(|| {
        refuse(
            iid_node,
            m,
            "iid sample length is not a statically-resolved 1-D count (dynamic, \
             multi-axis, or unresolved domain); only a 1-D static fan-out is built",
        )
    })?;
    let (ctor, kernel_input) = split_constructor(m, kernel).ok_or_else(|| {
        refuse(
            kernel,
            m,
            "iid sample: inner kernel must be a built-in constructor; a broadcast/array-of-kernels \
             form is not lowered here",
        )
    })?;
    Ok(build_iid_sample_term(
        m,
        ctor,
        kernel_input,
        n,
        rng,
        sample_name,
    ))
}

/// `iid(K, n)` → `(K, iid_node)`, resolving one level of ref indirection first
/// (mirroring `split_constructor`'s convention). The repeat count is read from
/// `iid_node`'s own inferred domain shape by `density::iid_static_size` — NOT
/// the raw `n` argument here — since a shape-dependent size (`lengthof(obs)`,
/// arithmetic, …) is already const-folded onto the type (see that function's
/// doc); callers pass `iid_node` straight to it.
fn split_iid(m: &Module, measure: NodeId) -> Option<(NodeId, NodeId)> {
    let (resolved, _) = resolve_ref_one(m, measure);
    let c = expect_builtin_call(m, resolved, "iid")?;
    (c.args.len() == 2).then_some((c.args[0], resolved))
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
    sample_name: Option<Symbol>,
) -> (NodeId, NodeId) {
    let sample = build_call(m, "builtin_sample", &[rng, ctor, kernel_input]);
    sample_slots(m, sample, sample_name)
}

/// Emit `builtin_sample(rng, ctor, kernel_input, n)` → `(get0(sample, 0)` = the
/// length-`n` IID array, `get0(sample, 1)` = new rng`)` — the spec §07
/// size-dims form of `builtin_sample`: ONE call over the fixed kernel
/// `ctor(kernel_input)` produces `n` iid draws and ONE advanced rngstate
/// (mirrors [`build_sample_term`]'s single-draw shape, plus the trailing `n`).
fn build_iid_sample_term(
    m: &mut Module,
    ctor: NodeId,
    kernel_input: NodeId,
    n: usize,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> (NodeId, NodeId) {
    let n_lit = m.alloc(Node::Lit(Scalar::Int(n as i64)));
    let sample = build_call(m, "builtin_sample", &[rng, ctor, kernel_input, n_lit]);
    sample_slots(m, sample, sample_name)
}

/// Bind a shared latent's primitive tuple before projecting either slot or
/// transforming its value, so every consumer reads the same draw by name.
fn sample_slots(m: &mut Module, sample: NodeId, name: Option<Symbol>) -> (NodeId, NodeId) {
    let sample = if let Some(name) = name {
        m.add_binding(Binding {
            name,
            rhs: sample,
            doc: None,
            public: false,
            synthetic: true,
        });
        m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }))
    } else {
        sample
    };
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
    // shared-ancestor identity (measure-algebra audit, 2026-06, H7/M4). Detect that and route
    // to the binding-rewrite path, which samples each latent once.
    let field_bids: Vec<Option<BindingId>> = fields
        .iter()
        .map(|&(_, v)| field_draw_binding(m, v))
        .collect();
    // A DERIVED field (`d = y1 - y2`) is not a variate at all — §13 "Output
    // reduction": "Other deterministic expressions pass through unchanged". It
    // still forces the binding-rewrite path whenever it reads latents, even if no
    // field is itself shared: the inline fold below would leave those latents as
    // un-lowered `draw` bindings for the derived expression to reference.
    let derived_latents = derived_field_latents(m, &fields, &field_bids);
    if requires_shared_binding_rewrite(m, &field_bids) || !derived_latents.is_empty() {
        return lower_shared_record_sample(m, &fields, &field_bids, &derived_latents, rng);
    }

    // Independent-draws fold (verified for >=2 independent draws): each field's
    // sample consumes the *previous* field's advanced rng (`cur = next`), not the
    // original `rng` re-read from scratch.
    let mut cur = rng;
    let mut out_fields = Vec::with_capacity(fields.len());
    for (name, val) in fields {
        // A derived field that reads NO latent (a constant, or an expression over
        // deterministic bindings only) needs nothing sampled — pass it through.
        // One that does read latents took the shared path above.
        if is_derived_value_field(m, val) {
            out_fields.push((name, val));
            continue;
        }
        // `val` is a `(%ref self <draw-binding>)` or an inline draw;
        // `lower_measure_sample` resolves either uniformly.
        let (v, next) = lower_measure_sample(m, val, cur)?;
        out_fields.push((name, v));
        cur = next;
    }
    Ok((build_record(m, &out_fields), cur))
}

/// Is this record field a DETERMINISTIC value expression — §13 "Output reduction":
/// "Other deterministic expressions pass through unchanged" — rather than a
/// variate or a measure?
///
/// §04's variate/measure distinction is what separates the three field kinds: a
/// `draw` (inline or via a binding) declares a variate and a `record` is a nested
/// measure product — both are [`lower_measure_sample`]'s business — while anything
/// whose inferred type is a confirmed VALUE type is an ordinary expression that
/// passes through, reading its sampled latents by name.
///
/// Fail-closed on the type: an unresolved/`%deferred` type is NOT confirmed, so a
/// field inference never typed keeps its current refusal instead of being passed
/// through as if deterministic. In particular an un-drawn measure in a field
/// (`record(a = Normal(…))`) is `Type::Measure` and so still refuses — admitting
/// the constructor leaf in measure position must not fabricate a draw here.
fn is_derived_value_field(m: &Module, value: NodeId) -> bool {
    let (resolved, _) = resolve_ref_one(m, value);
    if matches!(builtin_name(m, resolved), Some("draw") | Some("record")) {
        return false;
    }
    m.type_of(resolved).is_some_and(is_confirmed_value_type)
}

/// Is `ty` a confirmed VALUE type (§03 "Value types")? Excludes the measure-layer
/// types (`Measure`/`Kernel`/`Likelihood`), functions and modules, and every
/// unresolved form (`%deferred`/`%failed`/`%any`/type variable) — see
/// [`is_derived_value_field`] for why the unresolved cases must fail closed.
fn is_confirmed_value_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Scalar(_)
            | Type::Array { .. }
            | Type::TVector { .. }
            | Type::Record(_)
            | Type::Tuple(_)
            | Type::Table { .. }
    )
}

/// The draw-bindings read by the record's DERIVED value fields, in
/// first-encounter order.
///
/// These are latents the record's own field list may never name (`record(d = d)`
/// with `d = y1 - y2` names neither `y1` nor `y2`), but which must still be
/// sampled — once each, in dependency order — for the derived expression to read
/// them by name. They join the field seeds in [`lower_shared_record_sample`];
/// [`topo_draw_cone`]'s visited set dedupes the overlap.
fn derived_field_latents(
    m: &Module,
    fields: &[(Symbol, NodeId)],
    field_bids: &[Option<BindingId>],
) -> Vec<BindingId> {
    let mut found: Vec<BindingId> = Vec::new();
    for (&(_, val), &bid) in fields.iter().zip(field_bids) {
        if bid.is_some() || !is_derived_value_field(m, val) {
            continue;
        }
        let (resolved, _) = resolve_ref_one(m, val);
        for latent in transitive_draw_bindings(m, resolved) {
            if !found.contains(&latent) {
                found.push(latent);
            }
        }
    }
    found
}

/// The draw-bindings reachable from `root` THROUGH deterministic bindings, in
/// first-encounter order.
///
/// [`referenced_draw_bindings`] stops at the subtree it is handed, which is enough
/// for a draw's own kernel input — that names its latents directly. A derived
/// field sits one indirection further out (the field value is `(%ref self d)`,
/// and `d`'s RHS is what names `y1`/`y2`) and may chain (`e = d * 2.0`), so this
/// version follows a non-draw binding's RHS as well. A draw binding terminates the
/// walk: its own kernel-input latents are pulled in by [`topo_draw_cone`].
fn transitive_draw_bindings(m: &Module, root: NodeId) -> Vec<BindingId> {
    let mut found: Vec<BindingId> = Vec::new();
    let mut seen: Vec<BindingId> = Vec::new();
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
            && let Some(bid) = m.binding_by_name(*name)
            && !seen.contains(&bid)
        {
            seen.push(bid);
            if draw_argument(m, m.binding(bid).rhs).is_some() {
                found.push(bid);
            } else {
                queue.push(m.binding(bid).rhs);
            }
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    found
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
/// value (including any rebase transformation) replaces the latent's
/// `draw`-binding RHS and its advanced rng threads to the next latent. Consumers keep
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
    derived_latents: &[BindingId],
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    // Latents in dependency (topological) order: a latent is sampled after every
    // draw-binding its kernel input references (spec §07: thread one RNG state
    // sequentially in dependency order). The seeds are the fields' own latents plus
    // those only a DERIVED field reads (`derived_field_latents`), which the field
    // list may never name; `topo_draw_cone` dedupes the overlap.
    let mut seeds: Vec<BindingId> = field_bids.iter().flatten().copied().collect();
    seeds.extend(derived_latents.iter().copied());
    let cone = topo_draw_cone(m, &seeds);

    let mut cur = rng;
    for &bid in &cone {
        // Use the ordinary draw dispatcher, naming its primitive tuple before
        // any value transformation. Later latents resolve their parent's binding
        // to this sampled value; rebase and iid keep their usual batched draw.
        let sample_name = fresh_latent_name(m, bid, "sample");
        let (value, next) = lower_draw(m, m.binding(bid).rhs, cur, Some(sample_name))?;

        // Rewrite the latent's draw-BINDING to the sampled value; consumers keep
        // their `(%ref self <latent>)` and resolve to it by name. Recorded as
        // query-pinned for the same reason a density query's pin is
        // (`density::measure_reaches_draw`): the binding is no longer a `draw` and a
        // LATER density query must not read the realization as a model constant and
        // score the conditional at it.
        m.pin_binding_to_query_point(bid, value);

        // Thread the advanced rng from the SAME sample binding into the next latent.
        cur = next;
    }

    // Assemble the record. A field that references a (now-rewritten) latent keeps
    // its `(%ref self <latent>)` — the shared sample, read by name. So does a
    // DERIVED field: §13 "Other deterministic expressions pass through unchanged",
    // and its expression now reads the sampled latents by name, which is exactly
    // what makes it a deterministic function of the SAME draws rather than a
    // re-draw. Any other field (an inline draw, or a ref to a non-draw binding) is
    // sampled inline, threading the rng after the cone.
    let mut out_fields = Vec::with_capacity(fields.len());
    for (&(name, val), &bid_opt) in fields.iter().zip(field_bids) {
        if bid_opt.is_some() || is_derived_value_field(m, val) {
            out_fields.push((name, val));
        } else {
            let (v, next) = lower_measure_sample(m, val, cur)?;
            out_fields.push((name, v));
            cur = next;
        }
    }
    Ok((build_record(m, &out_fields), cur))
}

/// A fresh private name for a copied draw, its sample tuple, or a query's law,
/// following the parser's `__`-prefixed convention and deduped against bindings.
fn fresh_latent_name(m: &mut Module, latent: BindingId, kind: &str) -> Symbol {
    let latent_name = m.binding(latent).name;
    let base = m.resolve(latent_name).to_string();
    let mut candidate = format!("__{kind}_{base}");
    let mut n = 1;
    loop {
        let sym = m.intern(&candidate);
        if m.binding_by_name(sym).is_none() {
            return sym;
        }
        candidate = format!("__{kind}_{base}_{n}");
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
            && let Some(bid) = m.binding_by_name(*name)
            && draw_argument(m, m.binding(bid).rhs).is_some()
            && !found.contains(&bid)
        {
            found.push(bid);
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
