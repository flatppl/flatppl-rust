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
//!
//! **Two dispatchers, two admissible sets.** [`lower_closed_measure_sample`]
//! handles MEASURE position — `rand`'s second argument, which §07 types as "a
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
    Binding, BindingId, Module, NamedKind, Node, NodeId, Phase, Ref, RefNs, Scalar, Symbol, Type,
};

/// `rand(rng, m)` → a deterministic sample of the closed measure `m`.
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
/// call to. It is used ONLY to check [`rand_result_is_destructured`], which
/// dispatches the result shape (tuple vs bare value); see that function's doc
/// for why.
pub(crate) fn lower_rand(
    m: &mut Module,
    bid: BindingId,
    rand_node: NodeId,
) -> Result<NodeId, RefuseError> {
    let (rng, measure) = {
        let c = expect_builtin_call(m, rand_node, "rand")
            .ok_or_else(|| refuse(rand_node, m, "expected rand"))?;
        if c.args.len() != 2 {
            return Err(refuse(rand_node, m, "rand expects 2 args (rng, measure)"));
        }
        (c.args[0], c.args[1])
    };
    let (value, rng_out) = lower_closed_measure_sample(m, measure, rng)?;
    if rand_result_is_destructured(m, bid) {
        // Full spec §07 (value, new_rstate) contract: the caller destructures
        // both slots (or feeds s2 into another rand). Build the 2-tuple so the
        // parser's `get(_,1)`/`get(_,2)` (1-based) project value/rng.
        Ok(build_call(m, "tuple", &[value, rng_out]))
    } else {
        // Value-terminal shortcut: `draws = rand(...)` used as a bare value /
        // read by string selector — return the bare value (unchanged).
        Ok(value)
    }
}

/// Is `rand_bid`'s value DESTRUCTURED — read via an INTEGER-literal tuple
/// projection (`get(_, k)` / `get0(_, k)`) rather than used as a bare value?
///
/// `rand(rng, lawof(x))` infers to `Tuple([domain(x), RngState])` (spec §07;
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
/// [`referenced_draw_bindings`], `driver.rs`'s `subtree_contains_ref`).
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

/// Sample a MEASURE-POSITION measure — `rand`'s second argument, or the base of
/// a `pushfwd` — threading `rng`; returns `(value_node, advanced_rng_node)`.
///
/// §07 `rand(rstate, m)` types this argument as "a closed measure `m`", and §06
/// ("Fundamental measures and measure algebra") makes a primitive constructor
/// call a nullary kernel, i.e. closed; measure-to-measure operations preserve
/// arity, so `pushfwd(f, <closed>)` is closed too. §07's only stated exclusions
/// are "measures involving non-constant weighting … or multivariate truncation",
/// both still refused via [`classify_intractable_or_deferred`].
///
/// **Why this is a separate entry point from [`lower_measure_sample`].** That
/// function doubles as the per-FIELD dispatcher for a record law (both record
/// folds call it for every field, whatever the field's head), and the two
/// positions admit different sets — §04's distinction between a variate and a
/// measure. A record field is a declared variate or a deterministic expression;
/// `rand`'s argument is a closed measure. Putting the primitive-constructor leaf
/// case in the shared dispatcher would make `lawof(record(a = Normal(…)))`
/// silently *sample* a field the model never declared as a variate — fabricating
/// a draw (`tests/refuse.rs::undrawn_measure_in_a_record_field_still_refuses`
/// pins that it keeps refusing).
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
            // `lawof(?x)` itself infers to a DETERMINISTIC phase (spec §04 "Phase
            // of the reified law": lawof absorbs its argument's stochasticity
            // rather than propagating it — `crates/infer/src/ops.rs`'s `"lawof"`
            // phase arm traces `law_phase(?x)`, never `Phase::Stochastic`). So the
            // phase that matters here is `?x`'s own, not `lawof(?x)`'s: a `?x`
            // that is not Stochastic-phase (e.g. `lawof(3.0)`, or
            // `lawof(record(a = a))` where `a` is a plain constant, not a draw)
            // has no generative `draw` subgraph for `rand` to re-run — refuse
            // rather than silently echo the constant back out as a "sample".
            //
            // This check belongs to the `lawof` ARM, not to `rand` as a whole: a
            // bare `Normal(…)` or `pushfwd(f, Normal(…))` in measure position is
            // a measure, so its own phase is deterministic too, and a
            // rand-level phase check would refuse every composed measure.
            if !matches!(m.phase_of(inner), Some(Phase::Stochastic)) {
                return Err(refuse(
                    inner,
                    m,
                    "lawof's argument is not stochastic-phase (a Dirac/deterministic point) — \
                     rand samples the law of a STOCHASTIC subgraph, so lawof(<non-stochastic>) \
                     has no generative draw to sample; refuse rather than mislower",
                ));
            }
            lower_closed_measure_sample(m, inner, rng)
        }
        Some("pushfwd") => lower_pushfwd_sample(m, resolved, rng),
        // §07's own worked `rand` example is `rand(rstate, iid(Normal(0, 1), 10))`:
        // a fixed-kernel `iid` is a nullary kernel, hence closed. Routes to the
        // SAME [`lower_iid_sample`] as the `draw(iid(K, n))` spelling, so both emit
        // the identical batched term.
        Some("iid") => {
            let (kernel, iid_node) = split_iid(m, resolved)
                .ok_or_else(|| refuse(resolved, m, "iid expects 2 args (kernel, count)"))?;
            lower_iid_sample(m, kernel, iid_node, rng)
        }
        Some("record") => lower_record_of_draws_sample(m, resolved, rng),
        Some("draw") => lower_draw(m, resolved, rng),
        _ => {
            // The intractable/deferred set first — `weighted(w, M)` and friends
            // carry positional args, so `split_constructor` would reject them
            // anyway, but with the generic "not a constructor" message instead of
            // the actionable §07 one.
            if let Some(err) = classify_intractable_or_deferred(m, resolved) {
                return Err(err);
            }
            // The leaf: a primitive constructor call IS a closed measure (§06), so
            // `rand(s, Normal(mu = 0.0, sigma = 1.0))` samples it directly with no
            // `lawof` wrapper. This is also what terminates [`lower_pushfwd_sample`]'s
            // recursion on its base.
            if let Some((ctor, kernel_input)) = split_constructor(m, resolved) {
                return Ok(build_sample_term(m, ctor, kernel_input, rng));
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
/// This is the textbook easy direction: $X \sim M \Rightarrow f(X) \sim f_* M$ by
/// the definition of the pushforward (§06 `pushfwd`: $(f_* M)(Y) = M(f^{-1}(Y))$),
/// so sampling needs **no inverse and no Jacobian** — this path must never touch
/// `crate::invert` or the bijection registry the density side uses. `f` need not
/// even be injective. Nested `pushfwd`s work by the same recursion, and the base
/// case that terminates it is [`lower_closed_measure_sample`]'s
/// primitive-constructor leaf.
///
/// The forward application is emitted as `(%call f <sampled value>)`, which must
/// then actually resolve to FlatPDL — a surviving `%call` is neither a
/// deterministic op nor a `builtin_*` primitive. Two forms resolve, and this
/// function admits exactly those two:
///
/// * a **bare builtin** callee (`pushfwd(exp, …)`, where the map is a
///   [`Node::Const`]) — `canon::inline`'s `builtin_callee_head` rewrites the head
///   to a direct builtin call;
/// * a **reified** `functionof`/lambda map that beta-reduces under
///   [`crate::kernel::reduce_kernel_application`] — applied here rather than left
///   to the driver's fixpoint, so that a map which does NOT reduce is a refusal
///   instead of an unreduced `%call` emitted at exit 0.
///
/// The check matters because `is_flatpdl` is phase/type-based and does not flag a
/// surviving `CallHead::User`, so an unreduced application would leave the module
/// silently non-conformant.
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
    // A bare builtin map needs no reduction — the head rewrite handles it — EXCEPT
    // over a record variate, where the application is §04 auto-splatting against
    // that operator's own parameter names. This vertical does not resolve those, and
    // nothing downstream would catch the mismatch: `infer` reports no diagnostic for
    // `pushfwd(exp, lawof(record(y = …)))` and `is_flatpdl` is structural, so
    // `exp(record(y = …))` would pass every gate and fail only in the engine.
    if matches!(m.node(map), Node::Const(_)) {
        if expect_builtin_call(m, value, "record").is_some() {
            return Err(refuse(
                pushfwd_node,
                m,
                "pushfwd's map is a bare built-in operator and its base measure's variate is a \
                 record: applying it is §04 auto-splatting against the operator's own parameter \
                 names, which this vertical does not resolve — write the map as a functionof \
                 whose parameter names are the record's field names",
            ));
        }
        return Ok((build_user_call(m, map, value), rng_out));
    }
    // For a RECORD-valued variate the forward application is an auto-splatting
    // call, whose §04 correspondence rule the reducer does not enforce — check it
    // before reducing (see [`record_splat_mismatch`]).
    if let Some(kernel) = crate::kernel::resolve_reified(m, map) {
        if let Some(why) = record_splat_mismatch(m, value, &kernel.inputs) {
            return Err(refuse(
                pushfwd_node,
                m,
                &format!(
                    "pushfwd's map does not apply to the record variate of its base measure: \
                     {why}. §04 \"Calling conventions\" makes a record argument whose field \
                     names do not match the callable's argument names a static error, and \
                     binding only the parameters that DO match would silently project the \
                     variate onto those coordinates — refuse rather than drop the rest"
                ),
            ));
        }
    }
    let applied = build_user_call(m, map, value);
    let reduced = crate::kernel::reduce_kernel_application(m, applied).ok_or_else(|| {
        refuse(
            pushfwd_node,
            m,
            "pushfwd's map does not reduce when applied to the sampled variate: its parameter \
             list does not bind against the base measure's domain (for a record-valued base, \
             application binds the map's parameters by field name). The forward application \
             would survive as an unreduced call, which is not FlatPDL — refuse rather than \
             emit it",
        )
    })?;
    Ok((reduced, rng_out))
}

/// Do the map's parameter names correspond EXACTLY to the fields of a
/// record-valued variate? Returns `None` when they do, or when `value` is not a
/// record literal (there is nothing to splat); otherwise the offending name, for
/// the refusal message.
///
/// §04 "Calling conventions" (auto-splatting): "`f(record(a = x, b = y, ...))` …
/// [is] equivalent to `f(a = x, b = y, ...)`" and "A call with field or column
/// names that do not match the callable's argument names is a static error". The
/// forward application of a `pushfwd` map to a record variate is exactly such a
/// call. [`crate::kernel::reduce_kernel_application`]'s record branch binds each of
/// the callable's inputs from a like-named field and stops there — it never checks
/// that every FIELD was consumed — so a map declaring a strict SUBSET of the
/// record's fields reduces to a projection of the variate, silently dropping the
/// other coordinates. Since a marginalizing projection and a full transform are
/// different measures, that difference is a wrong sample, not a cosmetic one.
fn record_splat_mismatch(m: &Module, value: NodeId, inputs: &[(Symbol, Ref)]) -> Option<String> {
    let fields: Vec<Symbol> = {
        let c = expect_builtin_call(m, value, "record")?;
        c.named.iter().map(|n| n.name).collect()
    };
    let params: Vec<Symbol> = inputs.iter().map(|&(n, _)| n).collect();
    if let Some(f) = fields.iter().find(|f| !params.contains(f)) {
        return Some(format!(
            "variate field `{}` binds no map parameter",
            m.resolve(*f)
        ));
    }
    if let Some(p) = params.iter().find(|p| !fields.contains(p)) {
        return Some(format!(
            "map parameter `{}` names no variate field",
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
        Some("draw") => lower_draw(m, resolved, rng),
        // Intractable (outside rand's tractable set, spec §07) / deferred
        // (simply not built in this vertical) — see `classify_intractable_or_deferred`.
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
/// Two buckets, per spec §07's `rand` tractable set:
/// * **Intractable** — `weighted`/`logweighted`/`bayesupdate` (a reweighted
///   measure has no direct sampling recipe; realizing its law needs a
///   change-of-measure algorithm — rejection/importance sampling, MCMC — which
///   is out of scope for this MVP's exact, deterministic lowering), and a
///   `truncate` whose base is CONFIRMED multivariate (no general sampling
///   recipe for an arbitrary multivariate truncated region either).
/// * **Deferred** — `jointchain`/`kchain`/`superpose`, and a univariate
///   `truncate`: none of these are conceptually intractable (a later vertical
///   could add inverse-CDF/rejection truncated sampling, or thread the rng
///   through a Kleisli/joint chain), they are simply not built in this one
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
/// `d = draw(pushfwd(f, lawof(record(y1 = y1, y2 = y2))))` is a NEW draw,
/// independent of the existing `y1`/`y2`, not a spelling of `d = f(record(y1 =
/// y1, y2 = y2))`. Lowering it by reusing those samples would emit a joint that
/// is silently wrong in the opposite direction — perfectly dependent where the
/// model asked for independence. Drawing it correctly means sampling the base law
/// a SECOND time, independently of the surrounding model's copies of the same
/// latents, which this vertical does not build. The message points at the
/// deterministic spelling for the (common) case where that is what was meant.
fn refuse_draw_of_pushfwd(m: &Module, id: NodeId) -> RefuseError {
    refuse(
        id,
        m,
        "draw(pushfwd(f, M)) denotes a fresh, independent draw from the pushforward law, not a \
         deterministic function of the draws inside M — reusing those draws would emit a wrong \
         (perfectly dependent) joint, and drawing M a second time independently of the \
         surrounding model's own copies of its latents is not built in this vertical; if you \
         meant a deterministic function of the existing draws, write the field as that \
         expression instead",
    )
}

/// `weighted`/`logweighted`/`bayesupdate`: outside `rand`'s tractable set
/// (spec §07) — see [`classify_intractable_or_deferred`].
fn refuse_weighted_family(m: &Module, id: NodeId) -> RefuseError {
    refuse(
        id,
        m,
        "sampling a weighted/logweighted/bayesupdate measure is intractable (spec §07: \
         outside rand's tractable set — no direct sampling recipe for an arbitrary \
         reweighted measure) — refuse rather than mislower",
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
            "sampling a multivariate truncated measure is intractable (spec §07: outside \
             rand's tractable set — no general sampling recipe for an arbitrary multivariate \
             truncation) — refuse rather than mislower",
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
    let (inner_resolved, _) = resolve_ref_one(m, inner_measure);
    if let Some(err) = classify_drawn_measure(m, inner_resolved) {
        return Err(err);
    }
    // Fan-out: `draw(iid(K, n))` → ONE batched `builtin_sample`, via the same
    // [`lower_iid_sample`] the measure-position `rand(s, iid(K, n))` spelling uses.
    if let Some((kernel, iid_node)) = split_iid(m, inner_measure) {
        return lower_iid_sample(m, kernel, iid_node, rng);
    }
    let (ctor, kernel_input) = split_constructor(m, inner_measure).ok_or_else(|| {
        refuse(
            inner_measure,
            m,
            "sample leaf: expected a built-in kernel constructor",
        )
    })?;
    Ok(build_sample_term(m, ctor, kernel_input, rng))
}

/// `iid(K, n)` with a FIXED kernel `K` and a static length `n` → ONE batched
/// `builtin_sample(rng, ctor, input, n)` (spec §07 measure-eval-prims:
/// `builtin_sample`'s size-dims form returns an IID array `X` of size `n` with a
/// SINGLE advanced `new_rngstate`, not one per element).
///
/// Shared by BOTH spellings of the same measure, so they emit the identical term:
/// [`lower_draw`]'s `draw(iid(K, n))` and [`lower_closed_measure_sample`]'s
/// measure-position `rand(s, iid(K, n))` — §07's own worked `rand` example
/// (`random_data, rstate2 = rand(rstate, iid(Normal(0, 1), 10))`), admissible
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
            "iid sample: inner kernel must be a built-in constructor (a broadcast/\
             array-of-kernels measure has differing per-element params — not a \
             fixed-kernel fan-out; refuse rather than mislower)",
        )
    })?;
    Ok(build_iid_sample_term(m, ctor, kernel_input, n, rng))
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
) -> (NodeId, NodeId) {
    let sample = build_call(m, "builtin_sample", &[rng, ctor, kernel_input]);
    let zero = m.alloc(Node::Lit(Scalar::Int(0)));
    let one = m.alloc(Node::Lit(Scalar::Int(1)));
    let value = build_call(m, "get0", &[sample, zero]);
    let new_rng = build_call(m, "get0", &[sample, one]);
    (value, new_rng)
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
) -> (NodeId, NodeId) {
    let n_lit = m.alloc(Node::Lit(Scalar::Int(n as i64)));
    let sample = build_call(m, "builtin_sample", &[rng, ctor, kernel_input, n_lit]);
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
        {
            if let Some(bid) = m.binding_by_name(*name) {
                if !seen.contains(&bid) {
                    seen.push(bid);
                    if draw_argument(m, m.binding(bid).rhs).is_some() {
                        found.push(bid);
                    } else {
                        queue.push(m.binding(bid).rhs);
                    }
                }
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
        // Read the draw's inner measure BEFORE rewriting the binding (the measure
        // node is a distinct arena node from the `draw` binding RHS, so it survives
        // the rewrite; a later latent's `(%ref self mu)` resolves by name to the
        // now-sampled value).
        let measure = draw_argument(m, m.binding(bid).rhs)
            .ok_or_else(|| refuse(m.binding(bid).rhs, m, "shared-sample: expected a draw"))?;
        // Same classification `lower_draw` applies to its own inner measure, so a
        // shared latent drawing an intractable/deferred measure (or a `pushfwd`)
        // gets the ACTIONABLE reason rather than the generic "not a constructor"
        // one below.
        let (measure_resolved, _) = resolve_ref_one(m, measure);
        if let Some(err) = classify_drawn_measure(m, measure_resolved) {
            return Err(err);
        }
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
