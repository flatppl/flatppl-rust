//! Density disintegration — the independent-record, combinator, and primitive cases
//! (spec §06, "Density of composed measures").
//!
//! Entry point: [`lower_logdensityof`], which lowers a `logdensityof(lawof(M), v)` node
//! to a deterministic expression.
//!
//! ## Supported measure shapes
//!
//! **Independent record of draws** (Task 3):
//! ```text
//! logdensityof(lawof(record(a = draw(Mₐ), b = draw(M_b))), record(a = vₐ, b = v_b))
//!   ⤳  add(builtin_logdensityof(kₐ, inputₐ, vₐ), builtin_logdensityof(k_b, input_b, v_b))
//! ```
//!
//! **Measure combinators** (Task 4) — each wraps an inner measure; recursion bottoms out at a
//! primitive constructor:
//! - `weighted(w, M)` → `add(log(w(v)), density(M, v))` — the weight `w` may be a
//!   constant/scalar (used as-is) OR a **function of the variate**
//!   (§06 "Density of composed measures"), in
//!   which case it is **applied at the variate**: `log w(v)`. When `w`'s body
//!   contains measure ops (e.g. an inner `logdensityof`), the driver's subtree
//!   scan finds and lowers them on a later iteration and the plain-function call
//!   is beta-reduced by the backend.
//! - `logweighted(ℓ, M)` → `add(ℓ(v), density(M, v))` — likewise, `ℓ` may be a
//!   constant/scalar or a function of the variate, applied as `ℓ(v)` (already in
//!   log space, so no outer `log`).
//! - `superpose(M₁, …, Mₖ)` → `logsumexp([density(M₁, v), …, density(Mₖ, v)])`
//!   (§07 `logsumexp` takes a single real vector, not variadic scalars)
//! - `normalize(M)` → `density(M, v)` when `M` is already a probability measure
//!   (closed-form `logZ = 0`); when `M = truncate(base, interval(lo, hi))` with
//!   `base` a **normalized univariate continuous** constructor, → `sub(density(M, v),
//!   log(sub(touniform(base, hi), touniform(base, lo))))` — the closed-form
//!   `Z = CDF(hi) - CDF(lo)` via the `builtin_touniform` CDF transport
//!   (`builtin_touniform` is the CDF `F` only for univariate continuous kernels,
//!   §07 "Measure kernel evaluation primitives"; valid only for a univariate-continuous-normalized
//!   base — the transport is defined only for continuous built-in kernels and use
//!   of an undefined transport is a static error, §07 "Measure kernel evaluation primitives");
//!   when `M = logweighted(x → logdensityof(g2, x), g1)` is a pointwise product of
//!   two Gaussians (`g1`, `g2` both `Normal`), → `sub(add(density(g1, v),
//!   density(g2, v)), logZ)` with the Gaussian-overlap `logZ = density(Normal(mu =
//!   μ2, sigma = sqrt(σ1² + σ2²)), μ1)` (§08 Normal);
//!   **refuses** for any other `M` — a `truncate` whose `base` is unnormalized
//!   (e.g. `Lebesgue`, where the CDF-Z identity does not hold), or normalized but
//!   discrete (`Binomial`/`Poisson`/`Categorical`) or multivariate (`MvNormal`),
//!   for which `touniform` is undefined, and any `logweighted` that is not a
//!   Gaussian product (no closed-form mass rule; `totalmass` is OUT of FlatPDL).
//! - `truncate(M, S)` → `ifelse(in(v, S), density(M, v), neg(inf))` (the `_ in R`
//!   membership builtin, which infers to a boolean — `elementof` is a set-valued
//!   parameter declaration, not a membership predicate).
//! - `pushfwd(bijection(f, f_inv, logvol), M)` → `sub(density(M, f_inv(v)), logvol(f_inv(v)))`
//! - `iid(M, N)` → `Σ_{i<N} density(M, get0(v, i))` — **`N` is the static
//!   1-D repeat count read from the iid node's own const-evaluated domain
//!   shape** (`iid_static_size`), so a shape-dependent size (`iid(M,
//!   lengthof(obs))`, `sizeof(M)`, arithmetic on lengths, or a named/inline
//!   literal) resolves as readily as a raw literal — `flatppl_infer` (at
//!   `Level::Shape`) folds the size into that shape (static unroll; corpus `N`
//!   is small). A genuinely dynamic size, or a multi-axis / vector `size` (e.g.
//!   `iid(M, [2, 3])` — a valid §06 shape), is **refused** (the O(N) unroll
//!   handles only 1-D `N`; the vectorized broadcast+reduce over a multi-axis
//!   shape is the noted scale path, not built).
//! - `joint(M₁,…,Mₖ)` (**positional only**) → `Σᵢ density(Mᵢ, get0(v, i))` —
//!   **scalar-variate components only**; a component is accepted ONLY when its
//!   measure-domain kind is CONFIRMED `Scalar` — a component whose domain is
//!   confirmed non-scalar (e.g. `iid(Normal, 2)`) OR whose domain kind is
//!   unknown/`%deferred` (inference did not resolve it) is **refused up
//!   front**, fail-closed, by inspecting each component's measure-domain kind,
//!   NOT via the downstream recursive call (whose `get0(v, i)` value is
//!   `%deferred`/`%unknown`, so its domain guard is skipped). Keyword
//!   `joint(name = M, …)` (named components → record variate) shares the
//!   `joint` op name so it reaches the same dispatch arm, but its components
//!   live in `named` with an empty positional `args`, so it is **refused**
//!   rather than mislowered.
//!
//! **Likelihood query** (measure-algebra-audit.md H2): `logdensityof(likelihoodof(K, obs), θ)` is
//! handled at the `logdensityof` *entry* (not via `lower_measure_density`). Its
//! arg2 is the parameter point θ (a record), NOT the variate; the variate is the
//! `obs` baked into the likelihood. `K` is scored at `obs`, then each θ field is
//! inlined into THIS query's density subtree only — a per-query substitution of
//! `(%ref self <name>)` for the θ value (never a mutation of the shared module
//! binding, so two likelihood queries over the same params keep distinct θ
//! points) — §06 "Likelihood construction":
//! `densityof(likelihoodof(K, obs), θ) = pdf(κ(θ), obs)`.
//!
//! **Joint likelihood** (§06 "Combining likelihoods"):
//! `logdensityof(joint_likelihood(L1, …, Lk), θ)` = `Σᵢ logdensityof(Lᵢ, θ)` —
//! likelihoods combine by multiplying densities (summing log-densities), every
//! component scored at the SAME θ. Each `Lᵢ` is itself a likelihood, lowered by
//! recursing through the per-likelihood dispatch at the shared θ. Positional
//! components only (§06 form); a keyword `joint_likelihood` refuses.
//!
//! **Refused:** `kchain` marginals, keyword `joint`, keyword `joint_likelihood`,
//! `bayesupdate`, `disintegrate`,
//! `restrict`, `pushfwd` with a non-bijection argument, `iid` with a genuinely
//! dynamic size (not statically resolvable from its const-evaluated domain
//! shape) or a multi-axis / vector size, `joint` with a component whose measure-domain kind
//! is not CONFIRMED scalar (refused up front — a confirmed-non-scalar OR an
//! unknown/`%deferred` domain both refuse, fail-closed),
//! `normalize(truncate(base, …))`
//! whose `base` is not a univariate-continuous-normalized measure (an unnormalized
//! base, or a normalized-but-discrete/multivariate base — each with its own refuse
//! message), and any unrecognised shape.
//! (`likelihoodof` reaching `lower_measure_density` still refuses there as a
//! safety net — it is normally unwrapped at the `logdensityof` entry above.)

use crate::refuse::RefuseError;
use flatppl_core::{
    BindingId, Call, CallHead, Dim, Inputs, Mass, Module, NamedArg, NamedKind, Node, NodeId, Ref,
    RefNs, Scalar, ScalarType, Symbol, Type,
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower `logdensityof(lawof(M), v)` at `query` into a deterministic expression,
/// returning the new root node id. Refuses anything that cannot be structurally matched.
///
/// Side effect: each `draw` binding consumed by the density query is pinned to its
/// scored value (its binding's RHS is redirected to the pinned variate), so no
/// stochastic `draw` survives.
pub(crate) fn lower_logdensityof(m: &mut Module, query: NodeId) -> Result<NodeId, RefuseError> {
    let (arg1, arg2) = {
        let q = expect_builtin_call(m, query, "logdensityof")
            .ok_or_else(|| refuse(query, m, "expected logdensityof"))?;
        if q.args.len() != 2 {
            return Err(refuse(query, m, "logdensityof expects 2 args"));
        }
        (q.args[0], q.args[1])
    };
    // Likelihood query: arg2 is the PARAMETER point θ; the variate is the
    // observed data baked into the likelihood (§06 "densityof(likelihoodof(K,obs),θ)").
    // Both `likelihoodof` and `joint_likelihood` are likelihood-layer ops (each
    // typing to `Type::Likelihood`); dispatch on the op name too, since a
    // `(%ref self L)` to a likelihood binding may not carry the `Likelihood` type
    // on the ref node itself.
    let (resolved, _) = resolve_ref_one(m, arg1);
    if is_likelihood(m, arg1)
        || matches!(
            builtin_name(m, resolved),
            Some("likelihoodof") | Some("joint_likelihood")
        )
    {
        return lower_likelihood_density(m, resolved, arg2);
    }
    // Measure query: arg2 is the variate (existing path).
    let (measure_expr, v) = extract_logdensityof_args(m, query)?;
    lower_measure_density(m, measure_expr, v)
}

/// True iff `id` infers to a `Likelihood` type.
fn is_likelihood(m: &Module, id: NodeId) -> bool {
    matches!(m.type_of(id), Some(Type::Likelihood { .. }))
}

/// Lower `logdensityof(L, θ)` for a likelihood-layer `L`, dispatching on its op:
/// * `likelihoodof(K, obs)` → density of `K` at `obs`, θ inlined
///   ([`lower_likelihood_query`]);
/// * `joint_likelihood(L1, …, Lk)` → `Σᵢ logdensityof(Lᵢ, θ)`
///   ([`lower_joint_likelihood`]).
///
/// `resolved` is the likelihood node after one `(%ref self …)` hop. A
/// likelihood-typed node that is neither op (e.g. reached only via its type)
/// falls through to [`lower_likelihood_query`], which refuses unless it is a
/// well-formed `likelihoodof` (refuse-don't-mislower).
fn lower_likelihood_density(
    m: &mut Module,
    resolved: NodeId,
    theta: NodeId,
) -> Result<NodeId, RefuseError> {
    match builtin_name(m, resolved) {
        Some("joint_likelihood") => lower_joint_likelihood(m, resolved, theta),
        _ => lower_likelihood_query(m, resolved, theta),
    }
}

/// `logdensityof(joint_likelihood(L1, …, Lk), θ)` = `Σᵢ logdensityof(Lᵢ, θ)`
/// (§06 "Combining likelihoods": likelihoods combine by multiplying densities,
/// i.e. **summing log-densities** — `log L(θ) = Σᵢ log Lᵢ(θ)`), every component
/// scored at the SAME parameter point θ. Each component `Lᵢ` is itself a
/// likelihood — typically `likelihoodof(Kᵢ, obsᵢ)`, possibly bound by name, or a
/// nested `joint_likelihood` — so we recurse through [`lower_likelihood_density`]
/// at the shared θ (reusing the per-likelihood lowering, which inlines θ into
/// each component's own density subtree). A component that cannot be lowered
/// refuses (don't mislower).
///
/// **Positional only.** §06 spells `joint_likelihood(L1, L2, …)` as a positional
/// list; a keyword form has no §06 meaning, so a `joint_likelihood` carrying
/// named args is refused (consistent with how [`lower_joint`] treats a keyword
/// `joint`), rather than silently dropped.
fn lower_joint_likelihood(
    m: &mut Module,
    node: NodeId,
    theta: NodeId,
) -> Result<NodeId, RefuseError> {
    let components: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "joint_likelihood")
            .ok_or_else(|| refuse(node, m, "expected joint_likelihood"))?;
        if !c.named.is_empty() {
            return Err(refuse(
                node,
                m,
                "keyword joint_likelihood (named components) is not a §06 form",
            ));
        }
        if c.args.is_empty() {
            return Err(refuse(
                node,
                m,
                "joint_likelihood needs at least one component",
            ));
        }
        c.args.to_vec()
    };
    let mut terms = Vec::with_capacity(components.len());
    for comp in components {
        // Each component is a likelihood scored at the SHARED θ. Resolve one ref
        // hop and reuse the per-likelihood dispatch (also handles a nested
        // joint_likelihood).
        let (comp_resolved, _) = resolve_ref_one(m, comp);
        terms.push(lower_likelihood_density(m, comp_resolved, theta)?);
    }
    Ok(fold_add(m, &terms))
}

/// `logdensityof(likelihoodof(K, obs), θ)` = density of `K` at the observed `obs`,
/// with `K`'s free parameters bound from the θ record (§06 "Likelihood
/// construction", measure-algebra-audit.md H2).
///
/// Each θ field value is inlined into THIS query's emitted density subtree only
/// (a self-contained per-query substitution keyed on `(%ref self <name>)` — see
/// [`substitute_refs_by_name`]). We deliberately do NOT mutate the shared module
/// bindings for `<name>`: a second likelihood query over the same parameters must
/// score at ITS OWN θ point, not the last θ written globally. Leaving the
/// `mu = elementof(...)` param decls in place is valid FlatPDL (`is_flatpdl`
/// allows `elementof` parameter declarations); they become unused free params.
fn lower_likelihood_query(
    m: &mut Module,
    likelihoodof_node: NodeId,
    theta: NodeId,
) -> Result<NodeId, RefuseError> {
    let (k, obs) = {
        let c = expect_builtin_call(m, likelihoodof_node, "likelihoodof")
            .ok_or_else(|| refuse(likelihoodof_node, m, "expected likelihoodof"))?;
        if c.args.len() != 2 {
            return Err(refuse(
                likelihoodof_node,
                m,
                "likelihoodof expects 2 args (kernel, obs)",
            ));
        }
        (c.args[0], c.args[1])
    };
    let theta_map = theta_field_map(m, theta)?;
    let density = lower_measure_density(m, k, obs)?;
    // Refuse-don't-mislower: a θ param captured as a `functionof` / `kernelof`
    // reification *input* (a `(name, %ref self <name>)` boundary entry) cannot be
    // reached by the `substitute_refs_by_name` inliner below — `map_tree` walks
    // `children()`, which excludes a `Call`'s `Inputs` bucket (core `node.rs`,
    // `for_each_child`). Left un-inlined, that θ param stays a dangling `(%ref self
    // <name>)` inside the reification, so the density scores as a function of the
    // FREE param instead of at θ — a silent mislowering that still passes
    // `is_flatpdl`. This must hold in EVERY build profile (a `debug_assert` is
    // stripped in release), so we HARD REFUSE here rather than assert.
    if subtree_has_theta_capturing_input(m, density, &theta_map) {
        return Err(refuse(
            likelihoodof_node,
            m,
            "θ parameter captured inside a functionof/kernelof reification input cannot be \
             inlined per query; this density is not yet lowerable — refuse rather than mislower",
        ));
    }
    // Inline this query's θ values into ITS OWN density subtree: substitute each
    // `(%ref self <name>)` matching a θ field with that field's value node. No
    // shared binding is mutated, so sibling queries over the same params keep
    // their own θ points (fixes the cross-query parameter leak: two likelihood
    // queries over shared params would otherwise clobber each other's θ).
    Ok(substitute_refs_by_name(m, density, &theta_map))
}

/// Build the `name → θ-value` map from the θ record's `%field name = value`
/// entries. Refuses if θ is not a record, or names a parameter with no module
/// binding (a θ field that does not correspond to a declared param is a
/// mislowering hazard, not a valid point).
fn theta_field_map(m: &Module, theta: NodeId) -> Result<Vec<(Symbol, NodeId)>, RefuseError> {
    let (resolved, _) = resolve_ref_one(m, theta);
    let fields: Vec<(Symbol, NodeId)> = {
        let rec = expect_builtin_call(m, resolved, "record")
            .ok_or_else(|| refuse(theta, m, "logdensityof(L, θ): θ must be a record"))?;
        rec.named
            .iter()
            .filter(|n| n.kind == NamedKind::Field)
            .map(|n| (n.name, n.value))
            .collect()
    };
    for (name, value) in &fields {
        if m.binding_by_name(*name).is_none() {
            return Err(refuse(
                *value,
                m,
                "θ names a parameter with no module binding",
            ));
        }
    }
    Ok(fields)
}

/// Replace every `(%ref self <name>)` in the subtree at `root` with the node
/// mapped from `<name>` in `map`, returning the (possibly new) root node id.
///
/// Thin wrapper over the shared bottom-up rebuild [`crate::driver::map_tree`]
/// (which also backs `driver::substitute_in_tree`): both walk the same
/// `children()` enumeration and rebuild only-if-changed. This one keys on a
/// `Ref(SelfMod, name)` leaf rather than a target NodeId — so one pass inlines
/// ALL θ fields at once. A matched self-ref is a leaf, replaced wholesale before
/// its (nonexistent) children recurse.
///
/// **Scope limit (θ-capturing reification inputs).** `map_tree` walks
/// `children()`, which does NOT include a `Call`'s [`flatppl_core::Inputs`] — the
/// `(Symbol, Ref)` boundary entries of a `functionof` / `kernelof` reification.
/// Such an entry's `Ref` CAN be `Ref(SelfMod, name)`, so a θ param captured as a
/// reification boundary input would NOT be inlined by this walk (and an `Inputs`
/// slot cannot hold a value node anyway — it is a name reference). The caller
/// [`lower_likelihood_query`] therefore HARD REFUSES (in every build profile)
/// when [`subtree_has_theta_capturing_input`] reports such a capture, so this
/// walk is only ever reached for a density subtree free of θ-capturing inputs.
fn substitute_refs_by_name(m: &mut Module, root: NodeId, map: &[(Symbol, NodeId)]) -> NodeId {
    crate::driver::map_tree(m, root, &mut |m, id| {
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            if let Some((_, value)) = map.iter().find(|(n, _)| n == name) {
                return Some(*value);
            }
        }
        None
    })
}

/// True iff some `functionof` / `kernelof` reification reachable from `root`
/// carries a `Spec` [`flatppl_core::Inputs`] boundary entry whose `Ref` is a
/// `Ref(SelfMod, name)` for a `name` in `map` — i.e. a θ param captured as a
/// reification input that [`substitute_refs_by_name`]'s `children()`-only walk
/// would not reach. Backs the hard refuse in [`lower_likelihood_query`].
///
/// The scan is `%ref self`-aware: an emitted density subtree references its
/// weight/kernel reification by NAME (`(%call (%ref self w) v)`), not inline, so
/// the `functionof` carrying the θ-capturing input lives in `w`'s binding RHS,
/// one indirection away. `children()` alone (which stops at the `(%ref self w)`
/// leaf) would miss it, so whenever we meet a `Ref(SelfMod, name)` whose binding
/// RHS we have not yet visited, we descend into that RHS too. `visited_bindings`
/// bounds the walk against reference cycles.
fn subtree_has_theta_capturing_input(m: &Module, root: NodeId, map: &[(Symbol, NodeId)]) -> bool {
    let mut stack = vec![root];
    let mut visited_bindings = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        match m.node(id) {
            Node::Call(c) => {
                if let Some(flatppl_core::Inputs::Spec(entries)) = &c.inputs {
                    for (_, r) in entries.iter() {
                        if r.ns == RefNs::SelfMod && map.iter().any(|(n, _)| *n == r.name) {
                            return true;
                        }
                    }
                }
                m.for_each_child(id, |c| stack.push(c));
            }
            // Follow a self-ref into its binding RHS: the reification a density
            // subtree captures is bound by name, so it is not among `root`'s
            // `children()`. Guard against cycles via `visited_bindings`.
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => {
                if let Some(bid) = m.binding_by_name(*name) {
                    if visited_bindings.insert(bid) {
                        stack.push(m.binding(bid).rhs);
                    }
                }
            }
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Core recursive dispatcher
// ---------------------------------------------------------------------------

/// Compute the log-density of `measure_expr` at `v`, returning a deterministic node.
/// `measure_expr` may be a record-of-draws, a combinator, a `(%ref self x)` pointing
/// to one of those, or a bare primitive constructor.
pub(crate) fn lower_measure_density(
    m: &mut Module,
    measure_expr: NodeId,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    // Resolve a single level of `(%ref self x)` indirection on the measure side.
    let (measure_node, _binding_opt) = resolve_ref_one(m, measure_expr);

    // Dispatch on the measure op.
    let op = builtin_name(m, measure_node);

    match op {
        Some("record") => lower_record_of_draws(m, measure_node, v),
        Some("weighted") => lower_weighted(m, measure_node, v),
        Some("logweighted") => lower_logweighted(m, measure_node, v),
        Some("superpose") => lower_superpose(m, measure_node, v),
        Some("normalize") => lower_normalize(m, measure_node, v),
        Some("truncate") => lower_truncate(m, measure_node, v),
        Some("pushfwd") => lower_pushfwd(m, measure_node, v),
        // kchain marginal: discrete-finite latent → mass-weighted logsumexp;
        // continuous / infinite-discrete / non-enumerable → refuse (Task 5).
        Some("kchain") => crate::marginal::lower_kchain_marginal(m, measure_node, v),
        Some("iid") => lower_iid(m, measure_node, v),
        Some("broadcast") => lower_broadcast_kernel(m, measure_node, v),
        Some("joint") => lower_joint(m, measure_node, v),
        // A reified measure (`functionof` / `kernelof`) used AS a measure — its
        // body is the measure expression it reifies. Unwrap to the body and recurse
        // so a `broadcast(K, params)` body reaches the broadcast-kernel arm and a
        // bare constructor body reaches `build_density_term` (histfactory's
        // `functionof(Poisson.(expected))` scored via `likelihoodof`).
        Some("functionof") | Some("kernelof") => lower_reified_measure(m, measure_node, v),
        // Refused combinators — refused here rather than mis-lowered.
        // `likelihoodof` / `joint_likelihood` are normally unwrapped at the
        // `logdensityof` entry (their arg2 is θ, not a variate); reaching the
        // measure dispatcher means they were entered as a bare measure — refuse
        // (safety net) rather than emit `builtin_logdensityof(joint_likelihood, …)`.
        Some("markovchain")
        | Some("kscan")
        | Some("jointchain")
        | Some("bayesupdate")
        | Some("disintegrate")
        | Some("restrict")
        | Some("likelihoodof")
        | Some("joint_likelihood")
        | Some("locscale") => Err(refuse_op(measure_node, m)),
        // Fallthrough: treat as a primitive distribution constructor.
        _ => build_density_term(m, measure_node, v),
    }
}

// ---------------------------------------------------------------------------
// Record-of-independent-draws (Task 3)
// ---------------------------------------------------------------------------

/// One scored component of an independent product: the component measure node
/// (a `draw`'s argument, e.g. the `Normal(..)` constructor), the pinned variate
/// value node from `v`, and — when the component reached us through a binding
/// reference — that binding, so the driver can pin it to the scored value.
struct Component {
    /// The distribution-constructor (or combinator) node `mᵢ`.
    measure: NodeId,
    /// The matching part of `v` to score `mᵢ` at.
    pinned: NodeId,
    /// `Some(bid)` when the component is `(%ref self x)` pointing to a draw binding.
    draw_binding: Option<BindingId>,
}

/// Lower `record(a = draw(Mₐ), ...)` at `record_node` with value `v`.
fn lower_record_of_draws(
    m: &mut Module,
    record_node: NodeId,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    let components = match_independent_record(m, record_node, v)?;

    // Empty record (degenerate joint): a sum over no components. The independent-
    // product density is Σᵢ logdensityof(Mᵢ, xᵢ) (§06 "Density of composed measures"),
    // whose empty case is 0.
    if components.is_empty() {
        return Ok(m.alloc(Node::Lit(Scalar::Real(0.0))));
    }

    // Build density terms per component.
    let mut terms: Vec<NodeId> = Vec::with_capacity(components.len());
    for comp in &components {
        terms.push(lower_measure_density(m, comp.measure, comp.pinned)?);
    }

    // Pin each referenced draw binding to its scored value.
    for comp in &components {
        if let Some(bid) = comp.draw_binding {
            m.set_binding_rhs(bid, comp.pinned);
        }
    }

    Ok(fold_add(m, &terms))
}

/// Match `record(%field nameᵢ valueᵢ ...)` and pair each component with the
/// matching field of `v`. Returns one `Component` per field.
fn match_independent_record(
    m: &Module,
    record_node: NodeId,
    v: NodeId,
) -> Result<Vec<Component>, RefuseError> {
    let rec = expect_builtin_call(m, record_node, "record")
        .ok_or_else(|| refuse(record_node, m, "expected record"))?;
    if !rec.args.is_empty() {
        return Err(refuse(
            record_node,
            m,
            "record with positional args is not a field-keyed product",
        ));
    }

    let vrec = expect_builtin_call(m, v, "record")
        .ok_or_else(|| refuse(v, m, "value must be a record"))?;
    if !vrec.args.is_empty() {
        return Err(refuse(v, m, "value record with positional args"));
    }

    let mut components = Vec::with_capacity(rec.named.len());
    for field in rec.named.iter() {
        if field.kind != NamedKind::Field {
            return Err(refuse(
                record_node,
                m,
                "non-field named arg in measure record",
            ));
        }
        let pinned = lookup_field(m, &vrec.named, field.name)
            .ok_or_else(|| refuse(v, m, "missing field in value record"))?;

        let (measure, draw_binding) = resolve_component_draw(m, field.value).ok_or_else(|| {
            refuse(
                field.value,
                m,
                "field is not a draw or a reference to a draw",
            )
        })?;
        components.push(Component {
            measure,
            pinned,
            draw_binding,
        });
    }

    Ok(components)
}

/// Resolve a record-field value to its underlying draw's measure argument.
/// Returns `(measure_node, draw_binding)` where `draw_binding` is the binding
/// whose RHS is the `draw(...)`, if reached through a ref.
fn resolve_component_draw(m: &Module, value: NodeId) -> Option<(NodeId, Option<BindingId>)> {
    // Case A: `(%ref self x)` → look up binding `x`; its RHS must be `draw(mᵢ)`.
    if let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = m.node(value)
    {
        let bid = m.binding_by_name(*name)?;
        let rhs = m.binding(bid).rhs;
        let measure = draw_argument(m, rhs)?;
        return Some((measure, Some(bid)));
    }
    // Case B: inline `draw(mᵢ)` as the field value.
    if let Some(measure) = draw_argument(m, value) {
        return Some((measure, None));
    }
    None
}

// ---------------------------------------------------------------------------
// Combinator rules (Task 4)
// ---------------------------------------------------------------------------

/// Is `w_node` a **variate-dependent** weight — i.e. a function of the variate
/// rather than a constant/scalar (spec §06, "Density of composed measures":
/// `log densityof(weighted(w, M), x) = log w(x) + log densityof(M, x)`, where `w`
/// is a constant OR a function of the variate)?
///
/// The scalar/constant case lowers correctly with the weight node AS-IS (`log w`
/// / `lw`). A function-valued weight does NOT: it must be APPLIED to the variate
/// (`log w(v)` / `lw(v)`) before scoring. Emitting `log(w)` / `add(w, …)` on a
/// function object is a silent mislowering — and it *passes* `is_flatpdl` (the
/// weight is `Function`/`Kernel`-typed, not measure-typed), so the conformance
/// gate would not catch it. We therefore detect the function-valued case here so
/// [`lower_weighted`] / [`lower_logweighted`] apply the weight at the variate
/// (`build_user_call(m, w_node, v)`) rather than lowering it as a scalar.
///
/// Two surface shapes both reach us (dump-verified):
/// * inline `functionof(…)` / `kernelof(…)` reification — a builtin call whose
///   head is `functionof`/`kernelof` (carrying `inputs = Some(..)`);
/// * a reified callable bound by name — `(%ref self f)` whose inferred type is
///   `Type::Function` / `Type::Kernel`.
///
/// We catch both: resolve one ref level and test the reification head, and test
/// the inferred type of the (original) weight node. Either positive ⇒ refuse.
fn weight_is_variate_dependent(m: &Module, w_node: NodeId) -> bool {
    // Inferred type of the weight (catches the bound-by-name `(%ref self f)` form,
    // whose call head we never see): a reified callable types to Function/Kernel,
    // a constant/scalar weight does not.
    if matches!(
        m.type_of(w_node),
        Some(Type::Function { .. }) | Some(Type::Kernel { .. })
    ) {
        return true;
    }
    // Reification head (catches an inline `functionof(…)`/`kernelof(…)` weight even
    // if inference left its type `%deferred`); resolve one ref level so a named
    // reification is seen by its constructor.
    let (resolved, _) = resolve_ref_one(m, w_node);
    if matches!(
        builtin_name(m, resolved),
        Some("functionof") | Some("kernelof")
    ) {
        return true;
    }
    if matches!(
        m.type_of(resolved),
        Some(Type::Function { .. }) | Some(Type::Kernel { .. })
    ) {
        return true;
    }
    false
}

/// `logdensityof(weighted(w, M), v)` = `log w(v) + logdensityof(M, v)`, where `w`
/// is a constant/scalar OR a function of the variate (§06 "Density of composed measures").
/// A constant/scalar
/// weight is used as-is (`log(w) + density`); a variate-dependent (function)
/// weight is **applied at the variate** (`log w(v) + density`), with `w(v)` =
/// `build_user_call(m, w_node, v)` — see [`weight_is_variate_dependent`].
fn lower_weighted(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (w_node, m_inner) = {
        let c = expect_builtin_call(m, node, "weighted")
            .ok_or_else(|| refuse(node, m, "expected weighted"))?;
        if c.args.len() != 2 {
            return Err(refuse(node, m, "weighted expects 2 args"));
        }
        (c.args[0], c.args[1])
    };
    let inner_density = lower_measure_density(m, m_inner, v)?;
    // log w + density; a variate-dependent (function) weight is applied at v: log w(v).
    let w_scored = if weight_is_variate_dependent(m, w_node) {
        build_user_call(m, w_node, v)
    } else {
        w_node
    };
    let log_w = build_call(m, "log", &[w_scored]);
    Ok(build_call(m, "add", &[log_w, inner_density]))
}

/// `logdensityof(logweighted(ℓ, M), v)` = `ℓ(v) + logdensityof(M, v)`, where `ℓ`
/// is a constant/scalar OR a function of the variate (§06 "Density of composed measures").
/// The log-weight is
/// already in log space, so there is no outer `log`: a constant/scalar `ℓ` is used
/// as-is, and a variate-dependent (function) log-weight is **applied at the
/// variate** (`ℓ(v) + density`), with `ℓ(v)` = `build_user_call(m, lw_node, v)`.
fn lower_logweighted(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (lw_node, m_inner) = {
        let c = expect_builtin_call(m, node, "logweighted")
            .ok_or_else(|| refuse(node, m, "expected logweighted"))?;
        if c.args.len() != 2 {
            return Err(refuse(node, m, "logweighted expects 2 args"));
        }
        (c.args[0], c.args[1])
    };
    let inner_density = lower_measure_density(m, m_inner, v)?;
    let lw_scored = if weight_is_variate_dependent(m, lw_node) {
        build_user_call(m, lw_node, v)
    } else {
        lw_node
    };
    Ok(build_call(m, "add", &[lw_scored, inner_density]))
}

/// `logdensityof(superpose(M₁, …, Mₖ), v)` = `logsumexp([density(M₁,v), …, density(Mₖ,v)])`
fn lower_superpose(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    // Read the args list before any mutable borrow.
    let inner_measures: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "superpose")
            .ok_or_else(|| refuse(node, m, "expected superpose"))?;
        if c.args.len() < 2 {
            return Err(refuse(node, m, "superpose needs at least 2 components"));
        }
        c.args.to_vec()
    };

    let mut density_terms: Vec<NodeId> = Vec::with_capacity(inner_measures.len());
    for &mi in &inner_measures {
        density_terms.push(lower_measure_density(m, mi, v)?);
    }

    // §07 "Reductions and norms": `logsumexp(v)` takes a single real VECTOR, not
    // a variadic positional argument list — wrap the per-component densities in a
    // `vector` literal (`[t₁, …, tₖ]`) so the emitted call is `logsumexp([…])`.
    let terms_vec = build_call(m, "vector", &density_terms);
    Ok(build_call(m, "logsumexp", &[terms_vec]))
}

/// `logdensityof(normalize(M), v)` = `logdensityof(M, v) - logZ`, where
/// `Z = totalmass(M)` must be a **closed-form** deterministic expression — never
/// the `totalmass` query op, which is OUT of FlatPDL (measures are not values).
///
/// Two closed-form mass rules are available:
/// * If `M` is already a probability measure (`Type::Measure { mass:
///   Mass::Normalized, .. }`) then `Z = 1`, `logZ = 0`, so `normalize(M)` is
///   the identity on the density — it lowers to just `logdensityof(M, v)`.
/// * If `M` is `truncate(base, interval(lo, hi))` with `base` a **normalized
///   univariate continuous** probability measure (a primitive constructor such as
///   `Normal`, `domain = Scalar(Real)`), `Z = CDF(hi) - CDF(lo)` via the
///   `builtin_touniform` CDF transport (§06 "Density of composed measures", the
///   `normalize` rule `Z = totalmass(M)`): `logZ = log(touniform(base, hi) -
///   touniform(base, lo))`. The `-inf` outside-support gate is already handled by
///   the existing `truncate` lowering, so the emitted density term is just
///   `lower_measure_density(m, m_inner, v)` minus this `logZ`. The base must be
///   univariate-continuous-normalized AND a **leaf** built-in distribution
///   constructor (`Normal`, `Beta`, …), NOT a measure-combinator: the CDF-Z
///   identity holds only there, since `builtin_touniform` is the CDF `F` only for
///   univariate continuous kernels (§07 "Measure kernel evaluation primitives")
///   and the transport is defined only for continuous built-in kernels — use of
///   an undefined transport is a static error (§07 "Measure kernel evaluation
///   primitives"). A base that is NOT univariate-continuous-normalized, or IS
///   univariate-continuous-normalized but a composed combinator, does NOT take
///   this path:
///   - an unnormalized base (e.g. `Lebesgue(reals)`, true `Z = hi − lo`,
///     `touniform` undefined) falls through to the unnormalized refuse below;
///   - a normalized but DISCRETE base (`Binomial`/`Poisson`/`Categorical`,
///     `domain = Scalar(Integer)`) or MULTIVARIATE base (`MvNormal`,
///     `domain = Vector`) has no defined `touniform`, so it refuses with a
///     DISTINCT message (a discrete/multivariate truncation Z — e.g. a CMF /
///     finite-support sum — is a legitimate future follow-on, not an invalid
///     model), rather than mislowering to an undefined transport;
///   - a composed/`pushfwd` base (e.g. `pushfwd(exp_bijection, Normal(0,1))`, a
///     truncated log-normal) whose head is a measure-combinator, not a leaf
///     kernel, has no defined `touniform` head — it refuses with its OWN
///     leaf-constructor message ([`base_is_measure_combinator`]).
///
/// * If `M = logweighted(x → logdensityof(g2, x), g1)` is a POINTWISE PRODUCT OF
///   TWO GAUSSIANS (`g1`, `g2` both `Normal`), the normalizer is the Gaussian
///   overlap integral — itself a Gaussian (§08 Normal):
///   `Z = ∫ N(x; μ1, σ1)·N(x; μ2, σ2) dx = N(μ1; μ2, sqrt(σ1² + σ2²))`. So the
///   density is `logdensityof(g1, v) + logdensityof(g2, v) − logZ`, with
///   `logZ = logdensityof(Normal(mu = μ2, sigma = sqrt(σ1² + σ2²)), μ1)`
///   ([`recognize_gaussian_product`]). Only this exact shape takes the path —
///   any other `logweighted` base (a non-`Gaussian` factor, or ℓ scoring g2 at
///   something other than the reified argument) falls through to the refuse.
///
/// Any other unnormalized `M` (`Finite`, `LocallyFinite`, a non-truncate
/// combinator, a `logweighted` that is not a Gaussian product, …) has no
/// closed-form mass rule here, so we **refuse** rather than emit `totalmass`
/// (refuse-don't-mislower).
///
/// The emitted `−logZ` inherits §06's `Z ≠ 0` precondition ("`normalize` … with
/// `Z = totalmass(M)` finite and nonzero", §06 "Density of composed measures"): a
/// degenerate/empty interval collapses `Z → 0`, so `logZ → −∞`. That is the
/// backend's concern (a runtime `log(0)`), not statically checked here.
fn lower_normalize(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let m_inner = {
        let c = expect_builtin_call(m, node, "normalize")
            .ok_or_else(|| refuse(node, m, "expected normalize"))?;
        if c.args.len() != 1 {
            return Err(refuse(node, m, "normalize expects 1 arg"));
        }
        c.args[0]
    };

    // Read the inferred total-mass class of M. Resolve one level of ref
    // indirection so `m = Normal(...)` referenced by name is classified by the
    // constructor's mass, not the (typeless) ref node.
    let (m_inner_resolved, _) = resolve_ref_one(m, m_inner);
    let inner_mass = match m.type_of(m_inner_resolved) {
        Some(Type::Measure { mass, .. }) => Some(*mass),
        _ => None,
    };

    if inner_mass == Some(Mass::Normalized) {
        // Probability measure: Z = 1, logZ = 0. `normalize(M)` ≡ density(M, v).
        return lower_measure_density(m, m_inner, v);
    }

    // Closed-form Z for a truncated univariate measure: Z = CDF(hi) − CDF(lo),
    // via the builtin_touniform (CDF) transport (§06 "Density of composed
    // measures"). Extract the truncate/interval shape and its endpoints BEFORE any
    // mutable builds below (immutable reads of `m` must precede `m.alloc`/`build_*`
    // calls).
    //
    // `non_interval_truncation_set` distinguishes "this IS a `truncate(base, set)`
    // node, but `set` is not a literal `interval(lo, hi)` call" (e.g. a named set
    // like `posreals`/`nonnegreals`) from "the outer node is not a `truncate` at
    // all" — the former gets its OWN refuse message below (closed-form Z is only
    // implemented for a literal interval bound), rather than falling through to
    // the generic unnormalized-measure message, which would misleadingly imply
    // the base itself is the problem.
    let mut non_interval_truncation_set = false;
    let truncate_shape: Option<(NodeId, NodeId, NodeId)> = {
        if let Some(tc) = expect_builtin_call(m, m_inner_resolved, "truncate") {
            if tc.args.len() == 2 {
                let (base, set) = (tc.args[0], tc.args[1]);
                if let Some(ic) = expect_builtin_call(m, set, "interval") {
                    if ic.args.len() == 2 {
                        Some((base, ic.args[0], ic.args[1]))
                    } else {
                        None
                    }
                } else {
                    non_interval_truncation_set = true;
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    if non_interval_truncation_set {
        return Err(RefuseError {
            node,
            construct: "normalize".to_string(),
            reason: "normalize(truncate): closed-form Z is only implemented for an \
                     `interval(lo, hi)` truncation set; a named/other set is not yet supported"
                .to_string(),
        });
    }

    if let Some((base, lo, hi)) = truncate_shape {
        // The CDF-Z identity `Z = touniform(base, hi) − touniform(base, lo) =
        // totalmass(truncate(base, S))` (§06 "Density of composed measures", the
        // `normalize` rule `Z = totalmass(M)`) holds ONLY when `base` is a
        // normalized *univariate continuous* probability measure — `builtin_touniform`
        // is the CDF `F` only for univariate continuous kernels (§07 "Measure
        // kernel evaluation primitives") and the transport is defined only for
        // continuous built-in kernels; use of an undefined transport is a static
        // error (§07 "Measure kernel evaluation primitives"). So the base must be
        // BOTH `Mass::Normalized` AND have a real (continuous) SCALAR domain:
        // * an unnormalized base (e.g. `Lebesgue(reals)`) has true `Z = hi − lo`
        //   and no `touniform` — the CDF path would mislower;
        // * a normalized *discrete* base (`Binomial`/`Poisson`/`Categorical`,
        //   `domain = Scalar(Integer)`) or a normalized *multivariate* base
        //   (`MvNormal`, `domain = Vector`) has NO defined `touniform` transport,
        //   so `builtin_touniform(base, …)` is an undefined transport / silent
        //   mislowering that still passes `is_flatpdl`.
        // If the base is not univariate-continuous-normalized, do NOT take the
        // CDF-Z path — fall through to the refuse below (exactly as the
        // unnormalized `Lebesgue` base already does).
        let (base_resolved, _) = resolve_ref_one(m, base);
        let base_ty = m.type_of(base_resolved);
        let base_normalized = matches!(
            base_ty,
            Some(Type::Measure {
                mass: Mass::Normalized,
                ..
            })
        );
        let base_univariate_continuous = matches!(
            base_ty,
            Some(Type::Measure { domain, mass: Mass::Normalized })
                if matches!(**domain, Type::Scalar(ScalarType::Real))
        );
        // The type gate above is necessary but NOT sufficient: `kernel_and_input`
        // emits `builtin_touniform(<head>, …)` for ANY `CallHead::Builtin` head,
        // so a *composed* base whose head is a measure-combinator (`pushfwd`,
        // `weighted`, `superpose`, …) — e.g. a truncated log-normal
        // `pushfwd(exp_bijection, Normal(0,1))` — would emit
        // `builtin_touniform(pushfwd, …)`, an UNDEFINED transport (a static error,
        // §07 "Measure kernel evaluation primitives"). `touniform` is the CDF only
        // for a *leaf* built-in distribution kernel (`Normal`, `Beta`, …), never a
        // measure-algebra combinator. A combinator base may even infer to
        // `Scalar(Real)`, passing the type gate — so we must reject it on its HEAD,
        // not rely on downstream re-inference. Refuse with a message DISTINCT from
        // the discrete/multivariate one below.
        if base_univariate_continuous && base_is_measure_combinator(m, base_resolved) {
            return Err(RefuseError {
                node,
                construct: "normalize".to_string(),
                reason: "normalize(truncate): closed-form Z needs a leaf built-in distribution \
                         base; a composed/pushfwd base has no defined touniform transport"
                    .to_string(),
            });
        }
        if base_univariate_continuous {
            let density = lower_measure_density(m, m_inner, v)?; // truncate handles the -inf gate
            let (kernel, input) = kernel_and_input(m, base)?; // helper below
            let cdf_hi = build_touniform(m, kernel, input, hi);
            let cdf_lo = build_touniform(m, kernel, input, lo);
            let z = build_call(m, "sub", &[cdf_hi, cdf_lo]);
            let log_z = build_call(m, "log", &[z]);
            return Ok(build_call(m, "sub", &[density, log_z]));
        }
        if base_normalized {
            // Normalized but NOT univariate continuous — a discrete base
            // (`Binomial`/`Poisson`/`Categorical`, `domain = Scalar(Integer)`) or a
            // multivariate base (`MvNormal`, `domain = Vector`). This is a valid
            // model, but its closed-form Z is NOT `builtin_touniform(base, hi) −
            // touniform(base, lo)`: `touniform` is the CDF `F` only for a univariate
            // *continuous* kernel (§07 "Measure kernel evaluation primitives"), and use of the
            // transport on a non-continuous / multivariate kernel is a static error
            // there — plus `Z = F(hi) − F(lo)` is a univariate identity regardless.
            // Refuse with its OWN message (a discrete/multivariate truncation Z —
            // e.g. a CMF / finite-support sum — is a legitimate future follow-on),
            // distinct from the unnormalized-base refuse below.
            return Err(RefuseError {
                node,
                construct: "normalize".to_string(),
                reason: "normalize(truncate): closed-form Z via builtin_touniform is only \
                         defined for a univariate continuous base (touniform is the CDF only \
                         there, §07); a discrete or multivariate base needs a different \
                         closed-form Z (e.g. a CMF / finite-support sum), which is not yet \
                         implemented"
                    .to_string(),
            });
        }
    }

    // Closed-form Z for a POINTWISE PRODUCT OF TWO GAUSSIANS:
    // `normalize(logweighted(x → logdensityof(g2, x), g1))` with `g1`, `g2` both
    // `Normal` is `N(x; μ1, σ1)·N(x; μ2, σ2) / Z`, whose normalizer is the
    // Gaussian overlap integral — itself a Gaussian (§08 Normal):
    //   Z = ∫ N(x; μ1, σ1)·N(x; μ2, σ2) dx = N(μ1; μ2, sqrt(σ1² + σ2²)).
    // So `logdensityof(prod, v)` = logdensityof(g1, v) + logdensityof(g2, v) −
    // logZ, with logZ = logdensityof(Normal(mu = μ2, sigma = sqrt(σ1² + σ2²)), μ1)
    // (symmetric in g1/g2). The recognizer below matches ONLY this exact shape
    // (both factors `Normal`, ℓ a reified `x → logdensityof(g2, x)` scoring g2 AT
    // the variate); any OTHER `logweighted` base falls through to the refuse (a
    // non-Gaussian factor has no Gaussian-overlap Z — refuse-don't-mislower).
    if let Some(gp) = recognize_gaussian_product(m, m_inner_resolved) {
        // g1's density at the variate. `gp.g1` is the base's ORIGINAL (typed)
        // `Normal` constructor, so `build_density_term`'s domain-kind guard fires
        // if the variate is somehow not the Gaussians' scalar domain — the product
        // here is over a scalar variate, so a non-scalar `v` refuses (both factors
        // share `v`, so g1's guard suffices).
        let t1 = build_density_term(m, gp.g1, v)?;
        // g2's density at the variate. g2's factor is read as `(μ2, σ2)` value
        // nodes (the reified body may already be lowered to `builtin_logdensityof`,
        // so there is no g2 constructor node to reuse); rebuild the constructor.
        let g2_ctor = build_normal_ctor(m, gp.mu2, gp.sigma2);
        let t2 = build_density_term(m, g2_ctor, v)?;
        // logZ stddev = sqrt(add(pow(σ1, 2), pow(σ2, 2))): the overlap variance is
        // the SUM of the two variances (§08 Normal), not their difference.
        let two_a = m.alloc(Node::Lit(Scalar::Real(2.0)));
        let var1 = build_call(m, "pow", &[gp.sigma1, two_a]);
        let two_b = m.alloc(Node::Lit(Scalar::Real(2.0)));
        let var2 = build_call(m, "pow", &[gp.sigma2, two_b]);
        let var_sum = build_call(m, "add", &[var1, var2]);
        let overlap_sigma = build_call(m, "sqrt", &[var_sum]);
        // logZ = logdensityof(Normal(mu = μ2, sigma = overlap_sigma), μ1). The
        // synthetic constructor is freshly built (untyped) so `build_density_term`
        // scores it as a leaf Normal at the scalar point μ1.
        let overlap_normal = build_normal_ctor(m, gp.mu2, overlap_sigma);
        let log_z = build_density_term(m, overlap_normal, gp.mu1)?;
        let sum = build_call(m, "add", &[t1, t2]);
        return Ok(build_call(m, "sub", &[sum, log_z]));
    }

    // No closed-form mass rule for an unnormalized measure in this MVP.
    Err(RefuseError {
        node,
        construct: "normalize".to_string(),
        reason: "normalize of an unnormalized measure needs a closed-form mass rule; \
                 `totalmass` is not FlatPDL"
            .to_string(),
    })
}

/// A recognized `normalize(logweighted(…))` pointwise product of two Gaussians.
/// `g1` is the base's ORIGINAL (typed) `Normal` constructor node — scored at the
/// variate directly, so its domain-kind guard applies. The four `mu`/`sigma`
/// value nodes drive the closed-form overlap Z ([`lower_normalize`]); g2 is read
/// as loose `(μ2, σ2)` nodes rather than a constructor because the reified body
/// may already be lowered to `builtin_logdensityof` (no g2 constructor survives).
struct GaussianProduct {
    g1: NodeId,
    mu1: NodeId,
    sigma1: NodeId,
    mu2: NodeId,
    sigma2: NodeId,
}

/// Recognize `logweighted(ℓ, base)` as a pointwise product of two Gaussians:
/// `base` a `Normal` constructor (g1) and `ℓ` a reified `x → logdensityof(g2, x)`
/// with `g2` a `Normal` scored AT the reified argument `x`. Returns g1 and the
/// two factors' `mu`/`sigma` nodes, or `None` if the shape does not match (any
/// other `logweighted` base keeps [`lower_normalize`]'s refuse —
/// refuse-don't-mislower). Purely structural (`&Module`); the caller builds IR.
fn recognize_gaussian_product(m: &Module, logweighted: NodeId) -> Option<GaussianProduct> {
    let c = expect_builtin_call(m, logweighted, "logweighted")?;
    if c.args.len() != 2 {
        return None;
    }
    let (lw_node, base) = (c.args[0], c.args[1]);

    // Factor 1 (g1): the logweighted base is a `Normal` constructor.
    let (g1, g1_kwargs) = normal_ctor(m, base)?;
    let mu1 = find_kwarg(m, &g1_kwargs, "mu")?;
    let sigma1 = find_kwarg(m, &g1_kwargs, "sigma")?;

    // Factor 2 (g2): ℓ is a reified function `x → logdensityof(g2, x)` — a
    // `functionof` whose single positional body scores a `Normal` AT the reified
    // argument `x` (not at a constant, which would make ℓ a constant weight, not a
    // second Gaussian factor).
    let (lw_resolved, _) = resolve_ref_one(m, lw_node);
    let f = expect_builtin_call(m, lw_resolved, "functionof")?;
    if f.args.len() != 1 {
        return None;
    }
    // The reified callable's single input placeholder (`x`); g2 must be scored at
    // exactly this ref, so ℓ(x) = logdensityof(g2, x).
    let arg_ref = match &f.inputs {
        Some(Inputs::Spec(entries)) if entries.len() == 1 => entries[0].1,
        _ => return None,
    };
    let (mu2, sigma2) = gaussian_factor_scored_at(m, f.args[0], arg_ref)?;

    Some(GaussianProduct {
        g1,
        mu1,
        sigma1,
        mu2,
        sigma2,
    })
}

/// Read the `(mu, sigma)` of a `Normal` factor scored at the reified argument
/// `arg_ref`, from the reified body — which may be in EITHER form, since the
/// driver lowers the inner `logdensityof` independently and binding order does
/// not fix which fires first:
/// * not-yet-lowered `logdensityof(Normal(mu, sigma), arg)`, or
/// * already-lowered `builtin_logdensityof(Normal, record(mu, sigma), arg)`.
///
/// In both forms the point the factor is scored at must be exactly `arg_ref`
/// (the lambda argument) — scoring at anything else means ℓ is not the second
/// Gaussian factor of a product, so we return `None` and the caller refuses.
/// Likewise, `mu`/`sigma` themselves must NOT reference `arg_ref`: e.g.
/// `Normal(mu = x, sigma = 1.0)` scored at `x` is `N(x; x, 1)`, a constant, not a
/// second Gaussian *factor* of `x` — g1's params are checked outside the lambda,
/// so only g2 needs the guard.
fn gaussian_factor_scored_at(m: &Module, body: NodeId, arg_ref: Ref) -> Option<(NodeId, NodeId)> {
    // Not-yet-lowered: logdensityof(Normal(...), arg).
    if let Some(ld) = expect_builtin_call(m, body, "logdensityof") {
        if ld.args.len() != 2 || !is_ref_to(m, ld.args[1], arg_ref) {
            return None;
        }
        let (_g2, kwargs) = normal_ctor(m, ld.args[0])?;
        let mu2 = find_kwarg(m, &kwargs, "mu")?;
        let sigma2 = find_kwarg(m, &kwargs, "sigma")?;
        if references_ref(m, mu2, arg_ref) || references_ref(m, sigma2, arg_ref) {
            return None;
        }
        return Some((mu2, sigma2));
    }
    // Already-lowered: builtin_logdensityof(Normal, record(mu, sigma), arg).
    if let Some(bl) = expect_builtin_call(m, body, "builtin_logdensityof") {
        if bl.args.len() != 3
            || !is_normal_const(m, bl.args[0])
            || !is_ref_to(m, bl.args[2], arg_ref)
        {
            return None;
        }
        let mu2 = find_field(m, bl.args[1], "mu")?;
        let sigma2 = find_field(m, bl.args[1], "sigma")?;
        if references_ref(m, mu2, arg_ref) || references_ref(m, sigma2, arg_ref) {
            return None;
        }
        return Some((mu2, sigma2));
    }
    None
}

/// If `node` (after one ref hop) is a bare-kwarg `Normal(...)` constructor,
/// return `(constructor_node, kwargs)`; otherwise `None`.
fn normal_ctor(m: &Module, node: NodeId) -> Option<(NodeId, Vec<(Symbol, NodeId)>)> {
    let (resolved, _) = resolve_ref_one(m, node);
    let (sym, kwargs) = split_kernel_constructor(m, resolved)?;
    if m.resolve(sym) != "Normal" {
        return None;
    }
    Some((resolved, kwargs))
}

/// True iff `node` is exactly the reference `target` (the reified argument).
fn is_ref_to(m: &Module, node: NodeId, target: Ref) -> bool {
    matches!(m.node(node), Node::Ref(r) if *r == target)
}

/// Does the subtree rooted at `node` reference `target` anywhere (mirrors
/// `marginal::references_input`, but keyed on a `Ref` rather than a boundary-input
/// `Symbol`)? Used to reject a g2 `mu`/`sigma` that still depends on the lambda
/// argument — such a param is not a constant second Gaussian factor, it's an
/// expression of the variate, so the "product of two Gaussians" shape doesn't apply.
fn references_ref(m: &Module, node: NodeId, target: Ref) -> bool {
    is_ref_to(m, node, target)
        || m.node(node)
            .children()
            .into_iter()
            .any(|child| references_ref(m, child, target))
}

/// True iff `node` is the `Normal` kernel constant (a lowered `builtin_logdensityof`'s arg 0).
fn is_normal_const(m: &Module, node: NodeId) -> bool {
    matches!(m.node(node), Node::Const(sym) if m.resolve(*sym) == "Normal")
}

/// The value of the `name` keyword argument in a constructor's kwargs, if present.
fn find_kwarg(m: &Module, kwargs: &[(Symbol, NodeId)], name: &str) -> Option<NodeId> {
    kwargs
        .iter()
        .find(|(sym, _)| m.resolve(*sym) == name)
        .map(|&(_, value)| value)
}

/// The value of the `%field name` entry in a `record(...)` call node, if present.
fn find_field(m: &Module, record: NodeId, name: &str) -> Option<NodeId> {
    let Node::Call(c) = m.node(record) else {
        return None;
    };
    c.named
        .iter()
        .find(|n| n.kind == NamedKind::Field && m.resolve(n.name) == name)
        .map(|n| n.value)
}

/// Allocate a bare-kwarg `Normal(mu = mu, sigma = sigma)` constructor call — the
/// shape [`split_kernel_constructor`] / [`build_density_term`] consume.
fn build_normal_ctor(m: &mut Module, mu: NodeId, sigma: NodeId) -> NodeId {
    let mu_sym = m.intern("mu");
    let sigma_sym = m.intern("sigma");
    let normal_sym = m.intern("Normal");
    let named = vec![
        NamedArg {
            kind: NamedKind::Kwarg,
            name: mu_sym,
            value: mu,
        },
        NamedArg {
            kind: NamedKind::Kwarg,
            name: sigma_sym,
            value: sigma,
        },
    ];
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(normal_sym),
        args: Vec::<NodeId>::new().into(),
        named: named.into(),
        inputs: None,
    }))
}

/// The measure-algebra combinator / operator heads. A `truncate` base whose
/// builtin head is one of these is a COMPOSED measure, not a leaf distribution
/// constructor, so it has no defined `builtin_touniform` (CDF) transport — the
/// CDF-Z `normalize(truncate)` path must refuse it (composed/pushfwd-style
/// truncation base, see `normalize_truncate_pushfwd_base_refuses` in
/// `tests/refuse.rs`). Leaf constructors (`Normal`, `Beta`, …) are NOT in this
/// set, which is why membership is the cleanest "is this a combinator, not a
/// leaf kernel?" test (the constructor set is open-ended; the combinator
/// vocabulary is fixed, spec §06 measure algebra).
///
/// Cross-reference: `driver.rs::COMBINATOR_OPS` encodes the same measure-
/// combinator vocabulary for a DIFFERENT purpose (DCE eligibility of a
/// `Measure`-typed binding) and intentionally has different membership. A new
/// measure-algebra op may need adding to both lists — check `driver.rs` too.
pub(crate) const MEASURE_COMBINATOR_OPS: &[&str] = &[
    "pushfwd",
    "weighted",
    "logweighted",
    "superpose",
    "normalize",
    "truncate",
    "joint",
    "jointchain",
    "iid",
    "lawof",
    "draw",
    "kchain",
    "kscan",
    "markovchain",
    "locscale",
    "restrict",
    "bayesupdate",
    "disintegrate",
    "likelihoodof",
    "joint_likelihood",
];

/// True iff `base`'s builtin head is a measure-algebra combinator (in
/// [`MEASURE_COMBINATOR_OPS`]) rather than a leaf distribution constructor.
fn base_is_measure_combinator(m: &Module, base: NodeId) -> bool {
    matches!(builtin_name(m, base), Some(name) if MEASURE_COMBINATOR_OPS.contains(&name))
}

/// Extract `(kernel_const, kernel_input_record)` from a primitive constructor
/// `Normal(mu = …, sigma = …)` — the `builtin_*` primitive's arg 0/1 shape.
/// Resolves one level of `(%ref self x)` indirection first, so a truncation
/// base bound by name (`g = Normal(...); truncate(g, ...)`) is classified by
/// its constructor.
///
/// **Defense-in-depth: rejects positional constructor args.** This builds the
/// kernel input record from `c.named` only, exactly like [`build_density_term`]'s
/// primitive-constructor path — a positionally-written base (`Normal(0.0,
/// 1.0)`) has no `named` entries, so silently proceeding would emit
/// `builtin_touniform(Normal, record(), hi)`, a wrong (missing-parameter)
/// input record, rather than refusing. In the current call graph this cannot
/// actually fire: [`lower_normalize`] always lowers the base's own density
/// term via `build_density_term` first (which already refuses
/// `!c.args.is_empty()`) before reaching `kernel_and_input`, so a positional
/// base is refused upstream. But that ordering is not enforced by this
/// function's own signature, so the guard is repeated here rather than left
/// implicit/order-dependent on the caller.
fn kernel_and_input(m: &mut Module, ctor: NodeId) -> Result<(NodeId, NodeId), RefuseError> {
    let (ctor_resolved, _) = resolve_ref_one(m, ctor);
    let (ctor_sym, kwargs): (Symbol, Vec<(Symbol, NodeId)>) = {
        let Node::Call(c) = m.node(ctor_resolved) else {
            return Err(refuse(
                ctor_resolved,
                m,
                "truncation base must be a primitive constructor",
            ));
        };
        let CallHead::Builtin(sym) = c.head else {
            return Err(refuse(ctor_resolved, m, "non-builtin truncation base"));
        };
        if !c.args.is_empty() {
            return Err(refuse(
                ctor_resolved,
                m,
                "primitive constructor with positional args not supported",
            ));
        }
        (sym, c.named.iter().map(|n| (n.name, n.value)).collect())
    };
    let kernel = m.alloc(Node::Const(ctor_sym));
    let input = build_record(m, &kwargs);
    Ok((kernel, input))
}

/// `builtin_touniform(kernel, kernel_input, x)` — the CDF transport (§07).
fn build_touniform(m: &mut Module, kernel: NodeId, input: NodeId, x: NodeId) -> NodeId {
    let sym = m.intern("builtin_touniform");
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: vec![kernel, input, x].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }))
}

/// `logdensityof(truncate(M, S), v)` = `ifelse(in(v, S), logdensityof(M, v), neg(inf))`.
///
/// The gate is the `_ in R` membership builtin (FlatPIR head `in`), which infers
/// to a boolean — the spec's membership idiom (§06, `fn(_ in R)`). `elementof`
/// is a *set-valued parameter declaration* (`elementof(R)`), not a 2-arg
/// membership predicate, so it must not be used here (a 2-arg `elementof` infers
/// to `%deferred`, an ill-typed `ifelse` condition).
fn lower_truncate(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (m_inner, s_node) = {
        let c = expect_builtin_call(m, node, "truncate")
            .ok_or_else(|| refuse(node, m, "expected truncate"))?;
        if c.args.len() != 2 {
            return Err(refuse(node, m, "truncate expects 2 args (measure, set)"));
        }
        (c.args[0], c.args[1])
    };

    let inner_density = lower_measure_density(m, m_inner, v)?;
    let gate = build_call(m, "in", &[v, s_node]);
    let inf_sym = m.intern("inf");
    let inf_node = m.alloc(Node::Const(inf_sym));
    let neg_inf = build_call(m, "neg", &[inf_node]);
    Ok(build_call(m, "ifelse", &[gate, inner_density, neg_inf]))
}

/// `logdensityof(pushfwd(bij, M), v)` = `logdensityof(M, f_inv(v)) - logvol(f_inv(v))`
/// where `bij = bijection(f, f_inv, logvol)`.
///
/// Refuses if `bij` is not a `bijection(...)` node (directly or via one level of ref).
fn lower_pushfwd(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (bij_node, m_inner) = {
        let c = expect_builtin_call(m, node, "pushfwd")
            .ok_or_else(|| refuse(node, m, "expected pushfwd"))?;
        if c.args.len() != 2 {
            return Err(refuse(node, m, "pushfwd expects 2 args"));
        }
        (c.args[0], c.args[1])
    };

    // Resolve `bij_node` through one level of ref indirection.
    let (bij_resolved, _) = resolve_ref_one(m, bij_node);

    // Extract f_inv and logvol from the bijection node.
    let (f_inv_node, logvol_node) = {
        let bij = expect_builtin_call(m, bij_resolved, "bijection").ok_or_else(|| {
            refuse(
                bij_resolved,
                m,
                "pushfwd bijection arg must be a bijection(f, f_inv, logvol) node",
            )
        })?;
        if bij.args.len() != 3 {
            return Err(refuse(
                bij_resolved,
                m,
                "bijection expects 3 args (f, f_inv, logvol)",
            ));
        }
        (bij.args[1], bij.args[2])
    };

    // preimage = f_inv(v)
    let preimage = build_user_call(m, f_inv_node, v);
    // inner_density = logdensityof(M, preimage)
    let inner_density = lower_measure_density(m, m_inner, preimage)?;
    // logvol_val = logvol(preimage)
    let logvol_val = build_user_call(m, logvol_node, preimage);
    Ok(build_call(m, "sub", &[inner_density, logvol_val]))
}

/// Resolve the static repeat count `N` of an `iid(M, size)` from the iid node's
/// own INFERRED domain type — general over any const-resolvable size, not just a
/// raw literal.
///
/// `iid(M, size)` infers to `Measure { domain: Array { shape, elem }, .. }` where
/// `shape` is the const-evaluated `size` (`crates/infer/src/consteval.rs`): the
/// determiniser re-runs `flatppl_infer::infer` — which runs at `Level::Shape` —
/// each driver iteration, so a shape-dependent size (`lengthof(obs)`,
/// `sizeof(M)`, or fixed-value arithmetic over lengths — engine-concepts §17.1)
/// is already folded into a static `Dim` on the iid measure's domain. Reading it
/// here (rather than pattern-matching the raw size node) is the do-it-once,
/// reuse-existing-API mechanism: the const knowledge lives on the type, exposed
/// via [`Module::type_of`].
///
/// **Only a 1-D STATIC leading axis is accepted.** §06 "Independent composition"
/// admits an `iid` `size` as "an integer (1-D length) or a vector of positive
/// integers (multi-axis shape)":
/// * a genuinely dynamic size (`external`/runtime — `Dim::Dynamic`) is not
///   statically unrollable → `None` → refuse;
/// * a multi-axis / vector size (`iid(M, [2, 3])` → `shape.len() != 1`) is
///   refused → `None`: the O(N) static unroll in [`lower_iid`] handles only a
///   1-D `N`; the vectorized broadcast+reduce over a multi-axis shape is the
///   noted scale path, not built (conservative refuse-don't-mislower, not a bug).
///
/// A `%deferred` iid type (inference did not resolve `M`'s domain, so
/// `iid_type` returned `Type::Deferred`) also yields `None` — refuse, never
/// guess a size.
fn iid_static_size(m: &Module, iid_node: NodeId) -> Option<usize> {
    let Some(Type::Measure { domain, .. }) = m.type_of(iid_node) else {
        return None;
    };
    let Type::Array { shape, .. } = domain.as_ref() else {
        return None;
    };
    if shape.len() != 1 {
        return None; // multi-axis / vector size — not the 1-D unroll case
    }
    match shape[0] {
        Dim::Static(n) => Some(n as usize),
        Dim::Dynamic => None,
    }
}

/// `logdensityof(iid(M, N), v)` = `Σ_{i<N} logdensityof(M, get0(v, i))`
/// (§06 "Density of composed measures", "iid(M, n) → Σ_i log densityof(M, xᵢ)").
/// `N` is the static repeat count read from the iid node's own inferred domain
/// shape (see [`iid_static_size`]) — general over any const-resolvable size
/// (`lengthof(obs)`, `sizeof(M)`, arithmetic on lengths, a named or inline
/// literal), since `flatppl_infer` (at `Level::Shape`) has already folded the
/// size into that static shape. A genuinely dynamic size, or a multi-axis /
/// vector `size` (e.g. `[2, 3]`), is refused ([`iid_static_size`] returns
/// `None`). Static unroll (corpus N small; broadcast+reduce is the noted scale
/// path, not built).
/// `N == 0` is the empty independent product: Σ over an empty index set is 0, so
/// it lowers to the log-density literal `0.0` (consistent with the empty measure
/// `record()`), not a refusal.
///
/// **No scalar-`M` guard — deliberate asymmetry with [`lower_joint`].** `iid(M,
/// size)` is the product `M^⊗N` over ARRAYS of shape `size`, i.e. a NESTED variate
/// with a leading repeat axis `[N, …M-shape]` (§06 "Independent composition", the
/// `iid` bullet). So `get0(v, i)` recovers the full i-th
/// `M`-variate (an entire row), which is exactly what this rule scores `M` at —
/// correct for ANY `M`, scalar or not (a non-scalar `M`, e.g. `iid(MvNormal, n)`
/// or a nested `iid(iid(…), n)`, lowers correctly, its inner variate reached by a
/// further `get0`). `joint`, by contrast, has a HETEROGENEOUS variate: the flat
/// `cat` of its component variates, so `joint`'s positional `get0(v, i)` only
/// aligns when every component is scalar — hence the scalar-component guard in
/// [`lower_joint`]. Adding that guard here would WRONGLY refuse valid non-scalar
/// `iid`, so it is intentionally absent.
fn lower_iid(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    // The repeat count comes from the iid node's own const-evaluated domain shape
    // (see `iid_static_size`), so a `lengthof(obs)` / `sizeof(M)` / arithmetic
    // size resolves as readily as a raw literal. A genuinely dynamic or
    // multi-axis size yields `None` and refuses.
    let n = iid_static_size(m, node).ok_or_else(|| {
        refuse(
            node,
            m,
            "iid size is not a statically-resolved 1-D count (dynamic, multi-axis, \
             or unresolved domain); only a 1-D static size is unrolled",
        )
    })?;
    let m_inner = {
        let c =
            expect_builtin_call(m, node, "iid").ok_or_else(|| refuse(node, m, "expected iid"))?;
        if c.args.len() != 2 {
            return Err(refuse(node, m, "iid expects 2 args (measure, size)"));
        }
        c.args[0]
    };
    // Empty independent product: Σ over an empty index set is 0 (log-density 0),
    // exactly as an empty measure `record()` lowers (the iid Σ rule,
    // §06 "Density of composed measures", with an empty index set). Short-circuit
    // BEFORE `fold_add`, which requires at least one term.
    if n == 0 {
        return Ok(m.alloc(Node::Lit(Scalar::Real(0.0))));
    }
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        let idx = m.alloc(Node::Lit(Scalar::Int(i as i64)));
        let elem = build_call(m, "get0", &[v, idx]);
        terms.push(lower_measure_density(m, m_inner, elem)?);
    }
    Ok(fold_add(m, &terms))
}

/// `logdensityof(broadcast(K, arg0, arg1, …), obs)` where `K` is a distribution
/// **constructor** broadcast over per-cell parameter arrays — an array-of-kernels
/// measure (§04 broadcasting; its inferred type is `Measure{Array{shape, cell}}`).
/// Its density at the observed array `obs` is the sum over cells of the per-cell
/// kernel log-density: `Σ_cell logdensityof(K(paramsᵢ), obsᵢ)` (§06 "Density of
/// composed measures", the independent-product Σ rule lifted to an axis).
///
/// **Axis-native, no unroll.** The emitted density is a single axis-level
/// expression — identical for length-3 or length-3000, static or dynamic — so it
/// deliberately does NOT use the `get0`/`iid_static_size`/`fold_add` element
/// unrolling of [`lower_iid`]:
/// ```text
/// kernel_inputs = broadcast(record, p0 = arg0, p1 = arg1, …)   # array-of-records, per cell {pᵢ: argᵢ}
/// lp            = sum( broadcast(builtin_logdensityof, K, kernel_inputs, obs) )
/// ```
/// The positional data-args (`arg0, arg1, …`, everything after the head `K`) bind
/// to `K`'s ordered constructor parameter names (§08, via
/// [`flatppl_infer::distribution_param_names`]) IN ORDER; a data-arg passed by
/// keyword (`broadcast(K, mu = arr)`) keeps its given name. In the outer
/// broadcast, `builtin_logdensityof` is the broadcast head, `K` (the `Const`
/// constructor tag) rides along scalar, and `kernel_inputs` (array of records)
/// and `obs` (array) are zipped to per-cell log-densities; `sum` reduces to the
/// scalar joint. `broadcast`/`record`/`sum` are §04/§07 ops legal in FlatPDL.
///
/// **Head shapes.** The broadcast head resolves to a constructor name from either
/// a bare built-in `Const(sym)` (`broadcast(Poisson, …)`) or a standard-module
/// member `Ref { ns: Module(_), name }` (`broadcast(hepphys.ContinuedPoisson, …)`,
/// §09); both reduce to the member name and emit the BARE `Const(name)` kernel.
///
/// **Refuse-don't-mislower.** A value-broadcast (head a deterministic op like
/// `add`, or any other head shape — a `%ref self`/`%ref %local` binding, a
/// literal) is not a kernel — it is refused rather than treated as a measure. A
/// head whose resolved name is not a known distribution constructor
/// (`distribution_param_names` → `None`, e.g. a module *function* member) is
/// likewise refused.
fn lower_broadcast_kernel(
    m: &mut Module,
    measure: NodeId,
    obs: NodeId,
) -> Result<NodeId, RefuseError> {
    // Read the broadcast: args[0] is the head being broadcast; args[1..] the
    // positional data-args; `named` the keyword data-args.
    let (head, pos_args, kw_args) = {
        let c = expect_builtin_call(m, measure, "broadcast")
            .ok_or_else(|| refuse(measure, m, "expected broadcast"))?;
        let Some((&head, pos_args)) = c.args.split_first() else {
            return Err(refuse(measure, m, "broadcast has no head argument"));
        };
        let pos_args = pos_args.to_vec();
        let kw_args: Vec<(Symbol, NodeId)> = c.named.iter().map(|n| (n.name, n.value)).collect();
        (head, pos_args, kw_args)
    };

    // The head must resolve to a distribution CONSTRUCTOR NAME. Two shapes carry
    // one: a bare built-in `Node::Const(sym)` (`broadcast(Poisson, …)`), or a
    // standard-module member `Node::Ref { ns: Module(_), name }`
    // (`broadcast(hepphys.ContinuedPoisson, …)`, §09). Both reduce to the member
    // name symbol; the kernel is emitted BARE as `Const(name)` (below) — the same
    // tag the engine's kernel registry keys, module-qualified or not.
    //
    // A value-broadcast (`broadcast(add, a, b)`, head a deterministic op) reaching
    // the measure dispatcher must NOT be mislowered as a kernel; nor must a
    // non-module ref (`%ref self …` / `%ref %local …`, which are bindings, not
    // constructors). Refuse.
    let ctor_sym = match *m.node(head) {
        Node::Const(sym) => sym,
        Node::Ref(Ref {
            ns: RefNs::Module(_),
            name,
        }) => name,
        _ => {
            return Err(refuse(
                measure,
                m,
                "broadcast head is not a distribution constructor (value-broadcast used as a measure)",
            ));
        }
    };

    // Ordered constructor parameter names (spec §08). `None` ⇒ the head is not a
    // known distribution constructor (e.g. a deterministic op) ⇒ refuse.
    let param_names =
        flatppl_infer::distribution_param_names(m.resolve(ctor_sym)).ok_or_else(|| {
            refuse(
                measure,
                m,
                "broadcast head is not a known distribution constructor",
            )
        })?;

    // More positional data-args than the constructor has parameters cannot bind
    // by position — refuse rather than drop or misname.
    if pos_args.len() > param_names.len() {
        return Err(refuse(
            measure,
            m,
            "broadcast has more positional data-args than the constructor has parameters",
        ));
    }

    // Per-cell record fields: positional args bind to param names in order; a
    // keyword data-arg keeps its given name.
    let mut fields: Vec<(Symbol, NodeId)> = Vec::with_capacity(pos_args.len() + kw_args.len());
    for (i, &arg) in pos_args.iter().enumerate() {
        let name = m.intern(&param_names[i]);
        fields.push((name, arg));
    }
    fields.extend(kw_args);

    // kernel_inputs = broadcast(record, p0 = arg0, p1 = arg1, …): the `record`
    // constructor broadcast over the param arrays yields an array of per-cell
    // records. The field bindings are `Kwarg` on the broadcast call (each cell's
    // `record(pᵢ = argᵢ[cell])`).
    let broadcast_sym = m.intern("broadcast");
    let record_head = {
        let record_sym = m.intern("record");
        m.alloc(Node::Const(record_sym))
    };
    let record_kwargs: Vec<NamedArg> = fields
        .iter()
        .map(|&(name, value)| NamedArg {
            kind: NamedKind::Kwarg,
            name,
            value,
        })
        .collect();
    let kernel_inputs = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(broadcast_sym),
        args: vec![record_head].into(),
        named: record_kwargs.into(),
        inputs: None,
    }));

    // lp = sum(broadcast(builtin_logdensityof, K, kernel_inputs, obs)): the
    // `builtin_logdensityof` head is applied per cell to the scalar constructor
    // tag `K` and the zipped (kernel_input, obs) pair; `sum` reduces the array of
    // per-cell log-densities to the scalar joint density.
    let kernel = m.alloc(Node::Const(ctor_sym));
    let ldo_head = {
        let ldo_sym = m.intern("builtin_logdensityof");
        m.alloc(Node::Const(ldo_sym))
    };
    let per_cell = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(broadcast_sym),
        args: vec![ldo_head, kernel, kernel_inputs, obs].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }));
    Ok(build_call(m, "sum", &[per_cell]))
}

/// `logdensityof(joint(M₁,…,Mₖ), v)` = `Σ logdensityof(Mᵢ, get0(v, i))`
/// (§06 "Density of composed measures"). The variate is the positional `cat` of the
/// component variates.
///
/// **Scope:** positional `joint` only, scalar-variate components. `joint`'s
/// variate is the positional `cat` of the component variates
/// (§06 "Density of composed measures"); for
/// scalar-variate components the destructuring is `get0(v, i)`, one slot per
/// component. Component variates of higher rank need `cat`-slice routing, which
/// this does not build — a component whose measure domain is non-scalar
/// (e.g. `iid(Normal, 2)`, domain array[2]) is refused HERE, up front, by
/// inspecting each component's own measure domain kind. This is NOT left to the
/// downstream recursive call: `build_density_term`'s domain check compares the
/// measure domain against the value `get0(v, i)`, which infers to
/// `%deferred`/`%unknown`, so that guard would be skipped and the extra slots
/// silently dropped. Keyword `joint(name₁ = M₁, …)` (named components → record variate)
/// is out of scope: it shares the `joint` op name with the positional form, so
/// it does reach this function, but it carries its components in `named`
/// rather than `args`. That is checked explicitly, first, with its own
/// distinct refuse message — not left to fall through to the positional
/// arg-count guard, which would misname a keyword `joint` as merely
/// under-sized.
fn lower_joint(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let inner: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "joint")
            .ok_or_else(|| refuse(node, m, "expected joint"))?;
        if !c.named.is_empty() {
            return Err(refuse(
                node,
                m,
                "keyword joint (named components) is not yet lowered",
            ));
        }
        if c.args.len() < 2 {
            return Err(refuse(node, m, "joint needs at least 2 components"));
        }
        c.args.to_vec()
    };
    // Guard the scalar-only restriction HERE, before the recursion. `get0(v, i)`
    // assigns each component ONE scalar slot of the positional `cat` variate; a
    // non-scalar component (e.g. `iid(Normal, 2)`, domain array[2] → Vector) would
    // be silently mislowered — its extra slots dropped, `get0`-of-scalar nested
    // into a single slot. The downstream `build_density_term` domain check does
    // NOT catch this: `get0(v, i)` infers to `%deferred`/`%unknown`, so its guard
    // is skipped. The COMPONENT's own measure domain, by contrast, is known
    // (`iid(Normal,2)` → array[2]), so we can refuse it precisely here.
    //
    // **Fail-closed on an unknown/deferred component domain.** A component
    // whose measure domain inference has NOT resolved to `Some(Type::Measure {
    // domain, .. })` — so `component_kind` is `None` — is refused too, not
    // waved through. Only a component CONFIRMED `Some(VariateKind::Scalar)` is
    // accepted. Waving through the `None` case would repeat exactly the
    // mislowering hazard this guard exists to close: an untyped component's
    // true arity is unknown, so `get0(v, i)` might silently drop its extra
    // `cat` slots if it turns out to be non-scalar. Per refuse-don't-mislower,
    // "unconfirmed" and "confirmed non-scalar" both refuse; only "confirmed
    // scalar" lowers.
    for &mi in inner.iter() {
        let (mi_resolved, _) = resolve_ref_one(m, mi);
        let component_kind = match m.type_of(mi_resolved) {
            Some(Type::Measure { domain, .. }) => variate_kind(domain),
            _ => None,
        };
        if component_kind != Some(VariateKind::Scalar) {
            return Err(refuse(
                mi,
                m,
                "joint component variate kind is not confirmed scalar (unknown/deferred or \
                 non-scalar); refuse rather than mislower a cat-slice",
            ));
        }
    }
    let mut terms = Vec::with_capacity(inner.len());
    for (i, &mi) in inner.iter().enumerate() {
        let idx = m.alloc(Node::Lit(Scalar::Int(i as i64)));
        let elem = build_call(m, "get0", &[v, idx]);
        terms.push(lower_measure_density(m, mi, elem)?);
    }
    Ok(fold_add(m, &terms))
}

/// `functionof` / `kernelof` used AS a measure (spec §04 reification, §11 reified
/// callables). A reification reifies one body expression — the measure it stands
/// for (`broadcast(K, …)`, `Normal(…)`, a `record`-of-draws, …) — held as its
/// single positional `args[0]`; its `Inputs` boundary maps each input name to a
/// `Ref`. Scoring the reified measure is scoring its body, so we UNWRAP to `args[0]`
/// and recurse through the dispatcher: a `broadcast(K, params)` body then reaches
/// [`lower_broadcast_kernel`], a bare constructor body reaches [`build_density_term`]
/// (histfactory's `model = functionof(Poisson.(expected))` applied via
/// `likelihoodof` / `logdensityof(L, θ)`). This adds no new density emission — it
/// only routes the reified body to the existing arm.
///
/// **Boundary threading (refuse-don't-mislower).** Histfactory's boundary is a
/// SELF-REF (`(lam, %ref self lam)`): the body references the param as
/// `(%ref self lam)` — exactly what the per-query θ-inliner
/// ([`substitute_refs_by_name`]) rewrites — so unwrapping the wrapper and recursing
/// binds the θ point unchanged (an `Inputs::Auto` boundary is auto-traced to the
/// same self-refs). But a `Spec` entry whose `Ref` is a LOCAL placeholder
/// (`(x, %ref local _x_)` — a genuine lambda argument) is NOT θ-inlinable: the
/// SelfMod-keyed inliner cannot reach a `(%ref local _x_)` left in the body, and
/// once the wrapper is unwrapped the [`subtree_has_theta_capturing_input`] guard
/// (which scans for a SURVIVING reification's inputs) no longer sees the boundary.
/// Rather than emit a density with a dangling placeholder, we REFUSE.
fn lower_reified_measure(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let body = {
        let Node::Call(c) = m.node(node) else {
            return Err(refuse(node, m, "expected functionof/kernelof"));
        };
        if c.args.len() != 1 {
            return Err(refuse(
                node,
                m,
                "functionof/kernelof reifies exactly one body expression",
            ));
        }
        // A LOCAL placeholder boundary input is a genuine lambda argument the
        // per-query θ-inliner (SelfMod-keyed) cannot reach once the wrapper is
        // unwrapped — refuse rather than leave a dangling `(%ref local …)` in the
        // scored density.
        if let Some(flatppl_core::Inputs::Spec(entries)) = &c.inputs {
            if entries.iter().any(|(_, r)| r.ns == RefNs::Local) {
                return Err(refuse(
                    node,
                    m,
                    "functionof/kernelof used as a measure has a placeholder boundary input that \
                     cannot be inlined per query; this reified measure is not yet lowerable — \
                     refuse rather than mislower",
                ));
            }
        }
        c.args[0]
    };
    lower_measure_density(m, body, v)
}

// ---------------------------------------------------------------------------
// Primitive distribution constructor density term (Task 3 helper)
// ---------------------------------------------------------------------------

/// Build `builtin_logdensityof(kernel, kernel_input, pinned)` for a primitive
/// distribution constructor `measure` applied to keyword arguments.
/// The top-level structural kind of a variate type, for a conservative
/// domain/variate compatibility check. `None` for unknown (deferred / any /
/// type-var) or non-variate types — those never refuse. `Array` and `TVector`
/// share a kind, so a column-vs-row-vector annotation difference is not flagged.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VariateKind {
    Scalar,
    Vector,
    Record,
    Tuple,
    Table,
}

fn variate_kind(t: &Type) -> Option<VariateKind> {
    match t {
        Type::Scalar(_) => Some(VariateKind::Scalar),
        Type::Array { .. } | Type::TVector { .. } => Some(VariateKind::Vector),
        Type::Record(_) => Some(VariateKind::Record),
        Type::Tuple(_) => Some(VariateKind::Tuple),
        Type::Table { .. } => Some(VariateKind::Table),
        _ => None,
    }
}

/// Read a primitive constructor call `Ctor(kw1 = v1, kw2 = v2, ...)` into its
/// constructor symbol and keyword arguments. `None` if `node` is not a `Call`,
/// not builtin-headed, carries positional args, or has a non-kwarg named arg —
/// any of which means `node` is not a primitive distribution/kernel constructor
/// eligible for `builtin_logdensityof` / `builtin_sample`.
///
/// Shared by [`build_density_term`] (the density-side kernel/kernel_input build)
/// and the sample-side leaf (`sample::split_constructor`) — both need exactly
/// this constructor-symbol-plus-kwargs read before building their respective
/// `builtin_*` call.
pub(crate) fn split_kernel_constructor(
    m: &Module,
    node: NodeId,
) -> Option<(Symbol, Vec<(Symbol, NodeId)>)> {
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if !c.args.is_empty() {
        return None;
    }
    let mut kwargs = Vec::with_capacity(c.named.len());
    for n in c.named.iter() {
        if n.kind != NamedKind::Kwarg {
            return None;
        }
        kwargs.push((n.name, n.value));
    }
    Some((sym, kwargs))
}

pub(crate) fn build_density_term(
    m: &mut Module,
    measure: NodeId,
    pinned: NodeId,
) -> Result<NodeId, RefuseError> {
    // Refuse scoring a measure at a variate whose structural KIND clearly
    // mismatches the measure's variate domain — a scalar `Normal` scored at a
    // record / tuple / vector. Inference does not reject
    // this, so guard here per the refuse-don't-mislower discipline rather than
    // emit an ill-typed `builtin_logdensityof`. Conservative: an unknown side or
    // a matching kind passes; only a definite kind mismatch refuses. (The
    // determinizer recursion descends into record fields, so a nested mismatch
    // surfaces here as a scalar measure meeting a structured value.)
    let domain_kind = match m.type_of(measure) {
        Some(Type::Measure { domain, .. }) => variate_kind(domain),
        _ => None,
    };
    let obs_kind = m.type_of(pinned).and_then(variate_kind);
    if let (Some(dk), Some(ok)) = (domain_kind, obs_kind) {
        if dk != ok {
            return Err(refuse(
                pinned,
                m,
                "variate type does not match the measure's domain",
            ));
        }
    }

    let (ctor_sym, kwargs) = split_kernel_constructor(m, measure).ok_or_else(|| {
        refuse(
            measure,
            m,
            "primitive measure must be a built-in constructor call with only keyword arguments",
        )
    })?;

    let kernel = m.alloc(Node::Const(ctor_sym));
    let kernel_input = build_record(m, &kwargs);
    let builtin = m.intern("builtin_logdensityof");
    Ok(m.alloc(Node::Call(Call {
        head: CallHead::Builtin(builtin),
        args: vec![kernel, kernel_input, pinned].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    })))
}

// ---------------------------------------------------------------------------
// Helper: extract (measure_expr, v) from logdensityof(lawof(M), v)
// ---------------------------------------------------------------------------

/// Extract `(measure_expr, v)` from `logdensityof(measure, v)`.
///
/// The first argument is the measure whose density we score. It comes in two
/// shapes that both reduce to "the underlying measure":
///
/// * `lawof(M_value)` — `lawof` reifies a (stochastic) value to its law; we
///   score the value's law, i.e. `M_value` (a record-of-draws, a combinator,
///   …). This is the inline form the Task-3/4 record/combinator goldens use.
/// * a bare measure expression — e.g. `(%ref self pp)` where `pp = kchain(…)`
///   (or any combinator binding). Here the measure is already a measure; there
///   is no `lawof` wrapper to strip.
///
/// We resolve one level of ref indirection and strip a `lawof` if present;
/// otherwise we hand the (resolved) measure node straight to the dispatcher,
/// which classifies it by op.
fn extract_logdensityof_args(m: &Module, query: NodeId) -> Result<(NodeId, NodeId), RefuseError> {
    let q = expect_builtin_call(m, query, "logdensityof")
        .ok_or_else(|| refuse(query, m, "expected logdensityof"))?;
    if q.args.len() != 2 {
        return Err(refuse(query, m, "logdensityof expects 2 args"));
    }
    let measure_arg = q.args[0];
    let v = q.args[1];

    // Resolve a single level of `(%ref self x)` indirection so a measure bound by
    // name (`pp = kchain(…)`) is classified by its constructor, and a `lawof`
    // wrapper is visible whether inline or behind a ref.
    let (resolved, _) = resolve_ref_one(m, measure_arg);
    if let Some(law) = expect_builtin_call(m, resolved, "lawof") {
        if law.args.len() != 1 {
            return Err(refuse(resolved, m, "lawof expects 1 arg"));
        }
        return Ok((law.args[0], v));
    }
    // Bare measure expression: hand the original (unresolved) node to the
    // dispatcher, which itself resolves one ref level and dispatches by op.
    Ok((measure_arg, v))
}

// ---------------------------------------------------------------------------
// Utility: IR construction helpers
// ---------------------------------------------------------------------------

/// Allocate a positional builtin call `head(args…)`.
pub(crate) fn build_call(m: &mut Module, head: &str, args: &[NodeId]) -> NodeId {
    let sym = m.intern(head);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: args.to_vec().into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }))
}

/// Allocate a user-function call `(%call callee arg)`.
fn build_user_call(m: &mut Module, callee: NodeId, arg: NodeId) -> NodeId {
    m.alloc(Node::Call(Call {
        head: CallHead::User(callee),
        args: vec![arg].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }))
}

/// Allocate a `record(%field name value ...)` call from `(name, value)` pairs.
pub(crate) fn build_record(m: &mut Module, fields: &[(Symbol, NodeId)]) -> NodeId {
    let named: Vec<NamedArg> = fields
        .iter()
        .map(|&(name, value)| NamedArg {
            kind: NamedKind::Field,
            name,
            value,
        })
        .collect();
    let record = m.intern("record");
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(record),
        args: Vec::<NodeId>::new().into(),
        named: named.into(),
        inputs: None,
    }))
}

/// Combine density terms with `add`: a single term passes through; two or more
/// fold left into nested binary `add(acc, term)` calls.
fn fold_add(m: &mut Module, terms: &[NodeId]) -> NodeId {
    debug_assert!(!terms.is_empty(), "fold_add requires at least one term");
    let mut acc = terms[0];
    for &t in &terms[1..] {
        acc = build_call(m, "add", &[acc, t]);
    }
    acc
}

/// If `id` is `draw(mᵢ)`, return `mᵢ`; otherwise `None`.
pub(crate) fn draw_argument(m: &Module, id: NodeId) -> Option<NodeId> {
    let c = expect_builtin_call(m, id, "draw")?;
    if c.args.len() != 1 {
        return None;
    }
    Some(c.args[0])
}

/// The value of the `%field`-named entry `name` in `named`, if present.
fn lookup_field(_m: &Module, named: &[NamedArg], name: Symbol) -> Option<NodeId> {
    named
        .iter()
        .find(|n| n.kind == NamedKind::Field && n.name == name)
        .map(|n| n.value)
}

/// If `id` is a builtin call with head named `name`, return its [`Call`].
pub(crate) fn expect_builtin_call<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a Call> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) == name {
        Some(c)
    } else {
        None
    }
}

/// Return the builtin op name for `id`, or `None` if it is not a builtin call.
pub(crate) fn builtin_name(m: &Module, id: NodeId) -> Option<&str> {
    if let Node::Call(c) = m.node(id) {
        if let CallHead::Builtin(sym) = c.head {
            return Some(m.resolve(sym));
        }
    }
    None
}

/// Follow one level of `(%ref self x)` indirection: if `id` is a self-ref,
/// return `(binding.rhs, Some(bid))`; otherwise return `(id, None)`.
pub(crate) fn resolve_ref_one(m: &Module, id: NodeId) -> (NodeId, Option<BindingId>) {
    if let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = m.node(id)
    {
        if let Some(bid) = m.binding_by_name(*name) {
            return (m.binding(bid).rhs, Some(bid));
        }
    }
    (id, None)
}

/// A refusal naming the construct at `id`.
pub(crate) fn refuse(id: NodeId, m: &Module, reason: &str) -> RefuseError {
    let construct = match m.node(id) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(sym) => m.resolve(sym).to_string(),
            CallHead::User(_) => "user-call".to_string(),
        },
        other => format!("{other:?}"),
    };
    RefuseError {
        node: id,
        construct,
        reason: reason.to_string(),
    }
}

/// A refusal for an unhandled measure op (for the combinator refused-list).
pub(crate) fn refuse_op(id: NodeId, m: &Module) -> RefuseError {
    let op = builtin_name(m, id).unwrap_or("unknown").to_string();
    RefuseError {
        node: id,
        construct: op.clone(),
        reason: format!(
            "density lowering for `{op}` is not implemented (deferred to a later task)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct unit test of the leaf-constructor gate in `lower_normalize`
    /// (companion to `normalize_truncate_pushfwd_base_refuses` in
    /// `tests/refuse.rs`, which exercises the same gate black-box).
    ///
    /// Through the normal parse+infer path, inference classifies a composed base
    /// (`pushfwd(bij, Normal)`, `superpose(…)`, …) as `domain = %any` or
    /// `mass ≠ Normalized`, so it never reaches the `base_univariate_continuous`
    /// CDF-Z arm — the black-box `tests/refuse.rs` shape refuses via the
    /// discrete/multivariate arm instead. But the type gate is NOT a reliable
    /// guard against a composed head: if inference ever DID classify a `pushfwd`
    /// base as `Scalar(Real)` + `Normalized` it would pass the type gate and
    /// `kernel_and_input` would emit `builtin_touniform(pushfwd, …)`, an undefined
    /// transport (§07 "Measure kernel evaluation primitives"). We must reject on
    /// the HEAD, not rely on re-inference. This test builds exactly that adverse
    /// case by hand — a `pushfwd`-headed base FORCED to `Scalar(Real)` +
    /// `Normalized` — and asserts `lower_normalize` refuses with the DISTINCT
    /// leaf-constructor message (not the discrete/multivariate one).
    #[test]
    fn normalize_truncate_composed_head_refuses_leaf_constructor_message() {
        let mut m = Module::new();

        // A `pushfwd`-headed Call standing in for a composed base. Its two args
        // are inert leaves — the gate rejects it on the HEAD before touching them.
        let pushfwd_sym = m.intern("pushfwd");
        let a0 = m.alloc(Node::Lit(Scalar::Real(0.0)));
        let a1 = m.alloc(Node::Lit(Scalar::Real(0.0)));
        let base = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(pushfwd_sym),
            args: vec![a0, a1].into(),
            named: Vec::<NamedArg>::new().into(),
            inputs: None,
        }));
        // FORCE the adversarial classification the type gate would wave through.
        m.set_type(
            base,
            Type::Measure {
                domain: Box::new(Type::Scalar(ScalarType::Real)),
                mass: Mass::Normalized,
            },
        );

        // interval(1.0, 3.0)
        let lo = m.alloc(Node::Lit(Scalar::Real(1.0)));
        let hi = m.alloc(Node::Lit(Scalar::Real(3.0)));
        let interval = build_call(&mut m, "interval", &[lo, hi]);
        // truncate(base, interval) — typed non-normalized so it does not hit the
        // `inner_mass == Normalized` identity short-circuit.
        let truncate = build_call(&mut m, "truncate", &[base, interval]);
        m.set_type(
            truncate,
            Type::Measure {
                domain: Box::new(Type::Scalar(ScalarType::Real)),
                mass: Mass::Finite,
            },
        );
        let normalize = build_call(&mut m, "normalize", &[truncate]);
        let v = m.alloc(Node::Lit(Scalar::Real(2.0)));

        let err = lower_normalize(&mut m, normalize, v)
            .expect_err("composed (pushfwd) base must refuse, not emit builtin_touniform(pushfwd)");
        assert!(
            err.reason.contains("leaf built-in distribution"),
            "refusal names the leaf-constructor requirement: {err:?}"
        );
        assert!(
            err.reason.contains("composed") || err.reason.contains("pushfwd"),
            "refusal points at the composed/pushfwd base: {err:?}"
        );
        // DISTINCT from the discrete/multivariate refuse.
        assert!(
            !err.reason.contains("discrete") && !err.reason.contains("multivariate"),
            "leaf-constructor refuse must be distinct from the discrete/multivariate one: {err:?}"
        );
    }

    /// Drift guard between the two hand-maintained measure-op vocabulary
    /// lists: [`MEASURE_COMBINATOR_OPS`] here and `driver::COMBINATOR_OPS`.
    /// They intentionally have DIFFERENT membership (see the cross-reference
    /// comments on both), so this is deliberately NOT an equality check —
    /// it only guards the specific hazard the two comments warn about: "a new
    /// measure-algebra op may need adding to both lists — check the other
    /// file too" is easy to forget in practice.
    ///
    /// The op names [`lower_measure_density`] dispatches to a DEDICATED
    /// combinator-lowering rule (i.e. every match arm other than the
    /// `record`-of-draws case and the primitive-constructor fallthrough) are
    /// exactly the measure-algebra combinator vocabulary this module
    /// classifies elsewhere (`base_is_measure_combinator`) — so every one of
    /// them belongs in [`MEASURE_COMBINATOR_OPS`]. If a future combinator is
    /// added to the dispatch match in `lower_measure_density` but the author
    /// forgets to also add it to `MEASURE_COMBINATOR_OPS`, this test catches
    /// it (a composed base with that head would otherwise slip past
    /// `base_is_measure_combinator` and `kernel_and_input` would emit an
    /// undefined `builtin_touniform(<that-head>, …)` transport).
    #[test]
    fn measure_combinator_ops_covers_lower_measure_density_dispatch() {
        // Mirrors the non-fallthrough, non-`record` arms of
        // `lower_measure_density`'s match — kept in sync BY HAND with that
        // function; that is the drift this test exists to catch.
        const DISPATCHED_COMBINATOR_OPS: &[&str] = &[
            "weighted",
            "logweighted",
            "superpose",
            "normalize",
            "truncate",
            "pushfwd",
            "kchain",
            "iid",
            "joint",
            "markovchain",
            "kscan",
            "jointchain",
            "bayesupdate",
            "disintegrate",
            "restrict",
            "likelihoodof",
            "joint_likelihood",
            "locscale",
        ];
        for op in DISPATCHED_COMBINATOR_OPS {
            assert!(
                MEASURE_COMBINATOR_OPS.contains(op),
                "`{op}` is dispatched by lower_measure_density as a measure combinator \
                 but is missing from MEASURE_COMBINATOR_OPS — the two hand-maintained \
                 op-vocab lists have drifted; add `{op}` to MEASURE_COMBINATOR_OPS"
            );
        }
    }
}
