//! Density disintegration — the independent-record, combinator, and primitive cases
//! (spec §06, "Density of composed measures").
//!
//! Entry points: [`lower_logdensityof`], which lowers a `logdensityof(lawof(M), v)`
//! node to a deterministic expression, and [`lower_densityof`], which lowers the
//! plain-density form `densityof(lawof(M), v)` to `exp(<the same log-density
//! node>)` — FlatPPL has no separate `builtin_densityof` primitive (§07 lists six
//! `builtin_*` primitives, only one of them a density: `builtin_logdensityof`), so
//! `densityof` is defined as `exp(logdensityof(...))` (§06) and reuses the shared
//! [`lower_density_core`] dispatch rather than reimplementing it.
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
//!   when `M = superpose(weighted(w₁, A₁), …, weighted(wₖ, Aₖ))` is a convex
//!   superposition of normalized mixands (each `Aᵢ` a probability measure, each
//!   `wᵢ` a variate-independent scalar), → `sub(density(M, v), log(Σ wᵢ))` with
//!   the additive-mass normalizer `Z = Σ wᵢ · totalmass(Aᵢ) = Σ wᵢ` (§06
//!   "Additive superposition", "Density reweighting"); the weights-sum-to-one
//!   mixture idiom the spec names as canonical is the `Z = 1` (`log Σ wᵢ = 0`)
//!   special case;
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
//! - `joint(M₁,…,Mₖ)` (**positional**) → `Σᵢ density(Mᵢ, get0(v, i))` —
//!   **scalar-variate components only**; a component is accepted ONLY when its
//!   measure-domain kind is CONFIRMED `Scalar` — a component whose domain is
//!   confirmed non-scalar (e.g. `iid(Normal, 2)`) OR whose domain kind is
//!   unknown/`%deferred` (inference did not resolve it) is **refused up
//!   front**, fail-closed, by inspecting each component's measure-domain kind,
//!   NOT via the downstream recursive call (whose `get0(v, i)` value is
//!   `%deferred`/`%unknown`, so its domain guard is skipped).
//! - `joint(name₁ = M₁, …, nameₖ = Mₖ)` (**keyword/record**) → `Σᵢ
//!   density(Mᵢ, v.nameᵢ)` — the variate is a RECORD keyed by the SAME field
//!   names as the joint's components (§04 example, §06 "joint and iid
//!   (independent products)"); the value must itself be a `record(...)` node,
//!   and every field the joint names must be present in it (refuses
//!   otherwise — refuse-don't-mislower). Unlike positional `joint`, there is
//!   **no scalar-component restriction**: a record field may itself be any
//!   shape (vector, nested record, …) — the recursive
//!   `lower_measure_density`/`build_density_term` call already domain-checks
//!   each component against its own pinned field value, so no upfront kind
//!   guard is needed here the way it is for the flat `cat` slicing of the
//!   positional form. A `joint` call mixing BOTH positional and keyword
//!   components is neither form — **refused** rather than guessing which one
//!   was meant.
//!
//! **Likelihood query** (measure-algebra-audit.md H2): `logdensityof(likelihoodof(K, obs), θ)` is
//! handled at the `logdensityof` *entry* (not via `lower_measure_density`). Its
//! arg2 is the parameter point θ (a record), NOT the variate; the variate is the
//! `obs` baked into the likelihood. `K` is scored at `obs`, then each θ field is
//! inlined into THIS query's density subtree only — a per-query substitution of
//! `(%ref self <name>)` for the θ value, reaching THROUGH θ-dependent derived
//! bindings (a kernel param `a = f_a(theta2)` is inlined as `f_a(<θ.theta2>)`, so
//! a `builtin_logdensityof` term never dangles on an unbound `elementof` param) —
//! never a mutation of the shared module binding, so two likelihood queries over
//! the same params keep distinct θ points — §06 "Likelihood construction":
//! `densityof(likelihoodof(K, obs), θ) = pdf(κ(θ), obs)`.
//!
//! **Joint likelihood** (§06 "Combining likelihoods"):
//! `logdensityof(joint_likelihood(L1, …, Lk), θ)` = `Σᵢ logdensityof(Lᵢ, θ)` —
//! likelihoods combine by multiplying densities (summing log-densities), every
//! component scored at the SAME θ. Each `Lᵢ` is itself a likelihood, lowered by
//! recursing through the per-likelihood dispatch at the shared θ. Positional
//! components only (§06 form); a keyword `joint_likelihood` refuses.
//!
//! **Refused:** `kchain` marginals, keyword `joint_likelihood`,
//! `disintegrate`,
//! `pushfwd` with a non-bijection argument, `iid` with a genuinely
//! dynamic size (not statically resolvable from its const-evaluated domain
//! shape) or a multi-axis / vector size, positional `joint` with a component whose measure-domain kind
//! is not CONFIRMED scalar (refused up front — a confirmed-non-scalar OR an
//! unknown/`%deferred` domain both refuse, fail-closed), a keyword `joint` whose
//! value is not a record or is missing a named component's field, and any
//! `joint` mixing positional and keyword components,
//! `normalize(truncate(base, …))`
//! whose `base` is not a univariate-continuous-normalized measure (an unnormalized
//! base, or a normalized-but-discrete/multivariate base — each with its own refuse
//! message), and any unrecognised shape.
//! (`likelihoodof` and `bayesupdate` reaching `lower_measure_density` still
//! refuse there as a safety net — both are normally intercepted and lowered at
//! the `logdensityof` entry above: `bayesupdate(L, prior)` to
//! `logdensityof(L, θ) + logdensityof(prior, θ)`, §06 "Likelihoods and posteriors".
//! `restrict` likewise still refuses as a safety net here — it is normally
//! intercepted BEFORE the density query, at the driver's `restrict` scan, and
//! desugared into `bayesupdate(likelihoodof(kernel, x), marginal)` over the
//! disintegration on `x`'s field names, §06 "Measure restriction".)

use crate::refuse::RefuseError;
use flatppl_core::{
    BindingId, Call, CallHead, Dim, Inputs, Mass, Module, NamedArg, NamedKind, Node, NodeId, Ref,
    RefNs, Scalar, ScalarType, Symbol, Type,
};
use flatppl_infer::ModuleBundle;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower `logdensityof(lawof(M), v)` at `query` into a deterministic expression,
/// returning the new root node id. Refuses anything that cannot be structurally matched.
///
/// Side effect: each `draw` binding consumed by the density query is pinned to its
/// scored value (its binding's RHS is redirected to the pinned variate), so no
/// stochastic `draw` survives.
pub(crate) fn lower_logdensityof(
    m: &mut Module,
    query: NodeId,
    bundle: &ModuleBundle,
) -> Result<NodeId, RefuseError> {
    let (arg1, arg2) = parse_density_query_args(m, query, "logdensityof")?;
    lower_density_core(m, arg1, arg2, bundle)
}

/// Lower `densityof(M, x)` at `query` into a deterministic expression: the
/// PLAIN density, `exp(logdensityof(M, x))` (spec §06: `densityof` is the
/// density, `logdensityof` its log — FlatPPL has no separate `builtin_densityof`
/// primitive, §07's six `builtin_*` primitives include only the log form). Reuses
/// [`lower_density_core`] — the same dispatch [`lower_logdensityof`] uses — so
/// `densityof` refuses in EXACTLY the cases `logdensityof` would (the core's
/// `Err` propagates unchanged), and shares the same `draw`-pinning side effect.
pub(crate) fn lower_densityof(
    m: &mut Module,
    query: NodeId,
    bundle: &ModuleBundle,
) -> Result<NodeId, RefuseError> {
    let (arg1, arg2) = parse_density_query_args(m, query, "densityof")?;
    let log_density = lower_density_core(m, arg1, arg2, bundle)?;
    Ok(build_call(m, "exp", &[log_density]))
}

/// Parse `op(arg1, arg2)`'s two positional args at `query` (`op` is
/// `"logdensityof"` or `"densityof"`), refusing when `query` is not a
/// well-formed 2-arg call to `op`.
fn parse_density_query_args(
    m: &Module,
    query: NodeId,
    op: &str,
) -> Result<(NodeId, NodeId), RefuseError> {
    let q = expect_builtin_call(m, query, op)
        .ok_or_else(|| refuse(query, m, &format!("expected {op}")))?;
    if q.args.len() != 2 {
        return Err(refuse(query, m, &format!("{op} expects 2 args")));
    }
    Ok((q.args[0], q.args[1]))
}

/// Shared core behind [`lower_logdensityof`] / [`lower_densityof`]: dispatches
/// on the measure/likelihood-layer shape of `arg1` and returns the LOG-density
/// node scored at `arg2`. [`lower_logdensityof`] returns this unchanged;
/// [`lower_densityof`] wraps it in `exp` (§06:
/// `densityof(M,x) = exp(logdensityof(M,x))`).
fn lower_density_core(
    m: &mut Module,
    arg1: NodeId,
    arg2: NodeId,
    bundle: &ModuleBundle,
) -> Result<NodeId, RefuseError> {
    // A cross-module *direct query target* (a submodule likelihood handle
    // `m.L` or a bare measure handle `m.d`) is already grafted to a local host
    // node by the time this runs: the driver's `apply_rule` calls
    // [`graft_query_target`] on the SAME target first and only reaches this
    // function once that returned `Ok(None)` (same-module target — nothing to
    // graft), so `arg1` here is always already local. No graft happens in
    // this function.
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
        return lower_likelihood_density(m, resolved, arg2, bundle);
    }
    // Unnormalized-posterior query: `bayesupdate(L, prior)` scores its
    // log-likelihood term via [`lower_likelihood_density`], which needs the
    // `bundle` (a cross-module L in a posterior). Intercept it HERE, where the
    // bundle is in scope, rather than in the bundle-less [`lower_measure_density`]
    // dispatcher (which keeps a refuse arm for it as a safety net). arg2 is the
    // parameter point θ, shared by both the likelihood and the prior (§06
    // "Likelihoods and posteriors").
    if matches!(builtin_name(m, resolved), Some("bayesupdate")) {
        return lower_bayesupdate(m, resolved, arg2, bundle);
    }
    // Measure query: arg2 is the variate. Strip a `lawof` wrapper on the
    // (possibly grafted) measure node and hand it to the recursive dispatcher.
    let measure_expr = measure_of_arg(m, arg1)?;
    lower_measure_density(m, measure_expr, arg2)
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
    bundle: &ModuleBundle,
) -> Result<NodeId, RefuseError> {
    match builtin_name(m, resolved) {
        Some("joint_likelihood") => lower_joint_likelihood(m, resolved, theta, bundle),
        _ => lower_likelihood_query(m, resolved, theta, bundle),
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
    bundle: &ModuleBundle,
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
        terms.push(lower_likelihood_density(m, comp_resolved, theta, bundle)?);
    }
    Ok(fold_add(m, &terms))
}

/// `logdensityof(bayesupdate(L, prior), θ)` = `logdensityof(L, θ) +
/// logdensityof(prior, θ)` — the UNNORMALIZED posterior `dν(θ) = L(θ)·dπ(θ)`
/// (§06 "Likelihoods and posteriors": `bayesupdate(L, prior)` lowers to
/// `logweighted(fn(logdensityof(L, _)), prior)` — the prior reweighted by the
/// log-likelihood). This is the HMC inference target: the log-posterior up to the
/// (constant, dropped) evidence.
///
/// arg0 `L` is a likelihood-typed handle (`likelihoodof(K, obs)` /
/// `joint_likelihood`); arg1 `prior` is a measure (typically a keyword `joint`
/// over the same θ fields). Both are scored at the SAME parameter point θ (= `v`,
/// a record over the parameter fields):
/// * the **prior** via [`lower_measure_density`] — θ is its variate; and
/// * the **log-likelihood** via [`lower_likelihood_density`] — θ is its parameter
///   point, and each θ field is inlined into the likelihood's own density subtree.
///
/// The two log-densities are summed, mirroring [`lower_logweighted`]
/// (`add(weight_scored, inner_density)`) whose "weight" here is the log-likelihood
/// term. The emitted density is therefore two `builtin_logdensityof` terms (one
/// from L's kernel scored at `obs`, one from the prior scored at θ) under an
/// `add`.
///
/// **Refuse-don't-mislower:** a non-lowerable `prior` (e.g. one marginalizing an
/// internal continuous latent — a non-enumerable `kchain`) or a non-lowerable `L`
/// (an intractable kernel) propagates its `Err`, so the whole posterior refuses
/// rather than emit a partial density.
fn lower_bayesupdate(
    m: &mut Module,
    node: NodeId,
    v: NodeId,
    bundle: &ModuleBundle,
) -> Result<NodeId, RefuseError> {
    let (l_arg, prior_arg) = {
        let c = expect_builtin_call(m, node, "bayesupdate")
            .ok_or_else(|| refuse(node, m, "expected bayesupdate"))?;
        if c.args.len() != 2 {
            return Err(refuse(
                node,
                m,
                "bayesupdate expects 2 args (likelihood, prior)",
            ));
        }
        (c.args[0], c.args[1])
    };
    // Prior density: θ (= v) is the prior's variate. A non-lowerable prior
    // propagates its Err (refuse-don't-mislower).
    let prior_d = lower_measure_density(m, prior_arg, v)?;
    // Log-likelihood: θ (= v) is the likelihood's parameter point. Resolve one
    // `(%ref self L)` hop to the likelihood op (likelihoodof / joint_likelihood)
    // and reuse the shared per-likelihood lowering (which threads the bundle so a
    // cross-module kernel in the posterior also lowers).
    let l_d = lower_likelihood_density(m, resolve_ref_one(m, l_arg).0, v, bundle)?;
    // Unnormalized log-posterior = log-likelihood + log-prior.
    Ok(build_call(m, "add", &[l_d, prior_d]))
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
/// If `k` (a likelihood kernel argument) is a cross-module ref
/// `(%ref <alias> member)` — directly, or via one `(%ref self …)` hop — resolve
/// it against `bundle` and graft the referenced submodule kernel subtree into the
/// host, returning the grafted host node. A same-module `k` is returned
/// unchanged. The graft honors the `load_module` `%assign` load-time
/// substitutions (a substituted submodule parameter is replaced by the host
/// expression it was bound to).
///
/// **Refuses** (rather than mislowering) when: the cross-module ref is
/// unresolvable (dependency or member absent from the bundle); or a grafted
/// submodule dependency's name collides with an unrelated pre-existing host
/// binding (independent namespaces — reusing the host binding would score against
/// the wrong value).
///
/// The graft (see [`crate::crossmodule`]) also pulls the kernel's own parameter
/// bindings into the host, so the caller's `theta_field_map` and the density
/// lowering both run on a self-contained node with no further bundle access.
fn resolve_cross_module_kernel(
    m: &mut Module,
    bundle: &ModuleBundle,
    k: NodeId,
) -> Result<NodeId, RefuseError> {
    // Same graft path as the direct-query-target case; a same-module `k`
    // (`Ok(None)`) is returned unchanged.
    Ok(graft_cross_module_target(m, k, bundle)?.unwrap_or(k))
}

/// If `target` is — or resolves via one `(%ref self …)` hop to — a cross-module
/// `(%ref <alias> member)` into a loaded submodule, resolve it against `bundle`
/// and GRAFT the referenced submodule subtree into the host, returning
/// `Some(grafted_host_node)`. Returns `Ok(None)` when `target` is a same-module
/// (local) reference — the caller keeps its original node and dispatches
/// unchanged.
///
/// **Refuses** (`Err`, refuse-don't-mislower) when the cross-module ref is
/// unresolvable (dependency or member absent from the bundle), or when the graft
/// itself refuses (a grafted submodule dependency whose name collides with an
/// unrelated pre-existing host binding — see
/// [`crate::crossmodule::graft_subtree`]).
///
/// This is the ONE graft entry point shared by both cross-module cases:
/// * a likelihood KERNEL argument (`likelihoodof(m.kernel, obs)`) via
///   [`resolve_cross_module_kernel`], and
/// * a direct `logdensityof` query TARGET (`logdensityof(m.L, θ)` /
///   `logdensityof(m.d, v)`) via [`graft_query_target`], called by the driver
///   BEFORE [`lower_logdensityof`] ever runs on that query — so by the time
///   `lower_logdensityof` sees a query, its target is already local and does
///   no grafting of its own.
///
/// It does NOT reimplement grafting — it calls `crossmodule::resolve_module_ref`
/// then `graft_subtree` — so the `load_module` `%assign` load-time substitution
/// and the namespace-collision safety (§04 "Multi-file models") are preserved
/// unchanged.
fn graft_cross_module_target(
    m: &mut Module,
    target: NodeId,
    bundle: &ModuleBundle,
) -> Result<Option<NodeId>, RefuseError> {
    let module_ref = if is_module_ref(m, target) {
        target
    } else {
        let (candidate, _) = resolve_ref_one(m, target);
        if is_module_ref(m, candidate) {
            candidate
        } else {
            return Ok(None); // same-module target — no graft
        }
    };
    match crate::crossmodule::resolve_module_ref(bundle, m, module_ref) {
        Some(resolved) => crate::crossmodule::graft_subtree(m, &resolved, bundle)
            .map(Some)
            .map_err(|reason| refuse(module_ref, m, &reason)),
        None => Err(refuse(
            module_ref,
            m,
            "cross-module ref could not be resolved against the module bundle \
             (missing dependency or member); refuse rather than mislower",
        )),
    }
}

/// Defer-and-reloop support for a cross-module `logdensityof` query TARGET
/// (GAP D). If `query`'s measure target (arg 0) is — or resolves via one `(%ref
/// self …)` hop to — a cross-module `(%ref <alias> member)` into a loaded
/// submodule, graft the referenced subtree into the host and return
/// `Some(new_query)`: a freshly-built `logdensityof` whose target arg is the
/// LOCAL grafted node (arg 1, the variate / θ, is carried over unchanged).
///
/// The driver rewrites the binding to `new_query` and RETURNS WITHOUT lowering.
/// It then reloops, re-runs inference (typing the grafted subtree — crucially an
/// `iid`'s const-evaluated domain shape, which [`iid_static_size`] reads), and
/// re-scans: it finds this SAME query, now with a local, typed target, for which
/// this function returns `Ok(None)` (nothing to graft) so the driver lowers it
/// fully. Deferring the lowering to after a re-infer is what makes the grafted
/// `iid` size statically resolvable and the grafted (now-typed) dead kernel
/// binding sweepable — both were untyped when the old inline graft-then-lower
/// ran within a single call, before the next re-infer.
///
/// `Ok(None)`: same-module target — nothing grafted; the driver lowers normally.
/// `Err`: the cross-module ref is unresolvable, or the graft itself refuses (a
/// grafted submodule dependency colliding with an unrelated host binding —
/// refuse-don't-mislower).
///
/// Termination: a graft here fires AT MOST ONCE per query. It rewrites the
/// target from a `(%ref <alias> …)` (or a self-ref reaching one) to a local
/// grafted node, so on the next scan `graft_cross_module_target` sees a local
/// node and returns `Ok(None)` — the query then lowers (strictly reducing the
/// measure-node count) rather than grafting again.
pub(crate) fn graft_query_target(
    m: &mut Module,
    query: NodeId,
    bundle: &ModuleBundle,
) -> Result<Option<NodeId>, RefuseError> {
    // `query` may be either a `logdensityof(...)` or a `densityof(...)` — both
    // share this grafting logic; rebuild with the SAME op name the query
    // actually used (not a hard-coded `"logdensityof"`), so a grafted
    // `densityof` query stays a `densityof` for the driver's next scan.
    let op = density_query_op_name(m, query)?;
    let (arg1, arg2) = {
        let q = expect_builtin_call(m, query, op)
            .ok_or_else(|| refuse(query, m, &format!("expected {op}")))?;
        if q.args.len() != 2 {
            return Err(refuse(query, m, &format!("{op} expects 2 args")));
        }
        (q.args[0], q.args[1])
    };
    // Case 1: the target IS (or resolves via one `(%ref self …)` hop to) a
    // cross-module measure / likelihood HANDLE — graft it directly.
    if let Some(grafted) = graft_cross_module_target(m, arg1, bundle)? {
        return Ok(Some(build_call(m, op, &[grafted, arg2])));
    }
    // Case 2 (#194): the target is a cross-module kernel APPLICATION — a
    // `%call { head: User(callee), args: [input] }` whose CALLEE is a cross-module
    // ref (`logdensityof(m.k(input), pt)`). Graft the callee into the host and
    // rebuild the call with the grafted LOCAL callee, so the driver's re-infer
    // types the grafted kernel and `reduce_kernel_application` β-reduces the
    // now-local application next iteration.
    if let Some(new_call) = graft_kernel_application_callee(m, arg1, bundle)? {
        return Ok(Some(build_call(m, op, &[new_call, arg2])));
    }
    Ok(None) // same-module target — driver lowers normally
}

/// The density-query op name at `query` — `"logdensityof"` or `"densityof"` —
/// or a refusal when `query` is neither. Returns a `'static` string (not
/// borrowed from `m`) so callers can freely mix reads/writes of `m` afterward.
fn density_query_op_name(m: &Module, query: NodeId) -> Result<&'static str, RefuseError> {
    match builtin_name(m, query) {
        Some("logdensityof") => Ok("logdensityof"),
        Some("densityof") => Ok("densityof"),
        _ => Err(refuse(query, m, "expected logdensityof or densityof")),
    }
}

/// If the `logdensityof` target `arg1` is — or resolves via one `(%ref self …)`
/// hop to — a reified-kernel APPLICATION `%call { head: User(callee), args }`
/// whose `callee` (after one ref hop) is a cross-module `(%ref <alias> member)`
/// ref, GRAFT the callee subtree into the host (via [`graft_cross_module_target`],
/// which carries the load-time `%assign`, the host-collision refuse, and the
/// nested-ref handling), rebuild the `%call` with the grafted LOCAL callee as its
/// head, and return `Some(rebuilt_call)`.
///
/// The application's arguments (`args` / `named` / reification `inputs`) are
/// assumed host-local — only the callee is grafted — so they are carried over
/// unchanged. That assumption is checked, not merely asserted: any `args` /
/// `named` value that is itself (or resolves via one `(%ref self …)` hop to) a
/// cross-module ref is refused (see below) rather than spliced unresolved into
/// the rebuilt call.
///
/// The rebuilt call is returned to [`graft_query_target`], which wraps it in a
/// fresh `logdensityof` for the driver to substitute for this query's target.
/// The driver then defers (re-infers) and, on the next scan, sees a now-LOCAL
/// application whose callee is no longer a module ref — so this returns `Ok(None)`
/// and `reduce_kernel_application` β-reduces it (the GAP-D defer-and-reloop). The
/// callee going from a module-ref to a local grafted node bounds the graft to AT
/// MOST ONCE per query (termination).
///
/// `Ok(None)`: `arg1` is not a user-call application, or its callee is a LOCAL
/// (same-module) ref — the existing same-module `reduce_kernel_application`
/// handles that untouched (no regression).
///
/// `Err` (refuse-don't-mislower): the callee is an unresolvable / colliding
/// cross-module ref (or — downstream — grafts to a non-kernel) — the graft's own
/// `Err` propagates; OR one of the application's `args`/`named` values is
/// itself a cross-module ref (only the callee is grafted, so a cross-module
/// argument would splice a dangling, unresolvable ref into the kernel body —
/// a silent wrong density — rather than lower correctly).
fn graft_kernel_application_callee(
    m: &mut Module,
    arg1: NodeId,
    bundle: &ModuleBundle,
) -> Result<Option<NodeId>, RefuseError> {
    // The target may be inline (`logdensityof(m.k(input), pt)`) or bound by name
    // (`ka = m.k(input); logdensityof(ka, pt)`); resolve one ref hop to the call.
    let (call_node, _) = resolve_ref_one(m, arg1);
    let (callee, args, named, inputs) = {
        let Node::Call(c) = m.node(call_node) else {
            return Ok(None);
        };
        let CallHead::User(callee) = c.head else {
            return Ok(None); // a builtin-headed measure op, not a kernel application
        };
        (callee, c.args.clone(), c.named.clone(), c.inputs.clone())
    };
    // Only a CROSS-MODULE callee triggers the graft. A local callee (directly, or
    // via one `(%ref self …)` hop) is left for `reduce_kernel_application`.
    let (callee_resolved, _) = resolve_ref_one(m, callee);
    if !is_module_ref(m, callee) && !is_module_ref(m, callee_resolved) {
        return Ok(None);
    }
    // Refuse-don't-mislower (CRITICAL): only the CALLEE is grafted below; the
    // args / named values are carried over UNCHANGED into the rebuilt `%call`
    // (see the doc comment above — they are assumed host-local). That
    // assumption is false when an argument is itself — or resolves via one
    // `(%ref self …)` hop to — a cross-module ref (`logdensityof(m.k(m.rec),
    // pt)`): the raw unresolved `(%ref <alias> member)` would be spliced
    // unchanged into the rebuilt call and, from there, into the grafted
    // kernel body. `reduce_kernel_application` (structural) resolves only
    // `SelfMod` refs, so it cannot see through a cross-module one — the
    // dangling ref would survive lowering as a `Type::Failed("cross-module
    // resolution")`-tagged node standing in for a resolved value, a SILENT
    // WRONG DENSITY. Refuse instead; fully grafting a cross-module argument
    // (rather than just the callee) is a richer follow-up, not required here.
    let arg_is_cross_module =
        |id: NodeId| is_module_ref(m, id) || is_module_ref(m, resolve_ref_one(m, id).0);
    if let Some(&bad) = args.iter().find(|&&a| arg_is_cross_module(a)) {
        return Err(refuse(
            bad,
            m,
            "cross-module kernel application with a cross-module argument is not supported; \
             refuse rather than mislower a dangling reference",
        ));
    }
    if let Some(bad) = named.iter().find(|na| arg_is_cross_module(na.value)) {
        return Err(refuse(
            bad.value,
            m,
            "cross-module kernel application with a cross-module argument is not supported; \
             refuse rather than mislower a dangling reference",
        ));
    }
    // Graft the cross-module callee into the host. `graft_cross_module_target`
    // returns `None` only for a same-module ref (excluded by the guard above), so
    // treat that defensively as "nothing to graft".
    let Some(grafted_callee) = graft_cross_module_target(m, callee, bundle)? else {
        return Ok(None);
    };
    // Rebuild the `%call` with the grafted local callee as its head; the
    // host-local args / named / reification inputs carry over unchanged.
    Ok(Some(m.alloc(Node::Call(Call {
        head: CallHead::User(grafted_callee),
        args,
        named,
        inputs,
    }))))
}

/// True iff `id` is a `(%ref <alias> …)` cross-module reference.
fn is_module_ref(m: &Module, id: NodeId) -> bool {
    matches!(
        m.node(id),
        Node::Ref(Ref {
            ns: RefNs::Module(_),
            ..
        })
    )
}

fn lower_likelihood_query(
    m: &mut Module,
    likelihoodof_node: NodeId,
    theta: NodeId,
    bundle: &ModuleBundle,
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
    // Cross-module kernel ref (`(%ref <alias> member)` into a loaded submodule):
    // graft the referenced kernel subtree into the host so the density lowering
    // below runs on a self-contained node (spec §04 — a measure/kernel crosses
    // module boundaries freely). Grafting brings the kernel's own parameter
    // bindings into the host too, so `theta_field_map` (next line) can resolve
    // the θ field names. A same-module `k` is returned unchanged.
    let k = resolve_cross_module_kernel(m, bundle, k)?;
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
    // `(%ref self <name>)` matching a θ field with that field's value node, and
    // reach THROUGH θ-dependent derived bindings (`a = f_a(theta2)`) by inlining a
    // θ-substituted copy of their RHS. No shared binding is mutated, so sibling
    // queries over the same params keep their own θ points (fixes the cross-query
    // parameter leak: two likelihood queries over shared params would otherwise
    // clobber each other's θ), and the density becomes self-contained w.r.t. θ —
    // no `builtin_logdensityof` term is left reading an unbound `elementof` param.
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

/// Inline this query's θ into the density subtree at `root`, reaching THROUGH
/// θ-dependent derived bindings, and return the (possibly new) root node id.
///
/// Two leaf cases are inlined (everything else is left as a shared `(%ref self
/// …)`, so no non-θ binding is duplicated):
/// * a `(%ref self <name>)` where `<name>` is a θ FIELD → replaced by that field's
///   value node (the original per-query substitution); and
/// * a `(%ref self <x>)` where binding `x`'s RHS TRANSITIVELY depends on a θ param
///   (a realistic forward model's kernel params are DERIVED — `a = f_a(theta2)`,
///   `b = f_b(theta1, theta2)`) → replaced by a θ-inlined COPY of `x`'s RHS
///   ([`theta_inlined_binding`]), so e.g. `a` becomes `f_a(<θ.theta2>)`. Without
///   this, the density's `builtin_logdensityof(Normal, record(mu = a, sigma = b),
///   …)` would read `a`/`b`, whose definitions dangle on the unbound
///   `elementof(reals)` param bindings — an unevaluable ("no derivation for lp")
///   density, a silent failure.
///
/// A binding that is NOT θ-dependent (`f_a`, `c`, `observed_data`) stays a shared
/// `(%ref self …)` ref — those bindings survive in the FlatPDL and evaluate on
/// their own, so the density becomes self-contained w.r.t. θ WITHOUT mutating any
/// shared binding (leak-free: two `logdensityof` queries over the same params keep
/// distinct θ points — a global pin on `a`/`b` would clobber one with the other).
///
/// **Scope limit (θ-capturing reification inputs).** [`crate::driver::map_tree`]
/// walks `children()`, which does NOT include a `Call`'s [`flatppl_core::Inputs`] —
/// the `(Symbol, Ref)` boundary entries of a `functionof` / `kernelof`
/// reification. A θ param captured there would NOT be inlined by this walk (and an
/// `Inputs` slot cannot hold a value node anyway). The caller
/// [`lower_likelihood_query`] therefore HARD REFUSES (in every build profile,
/// BEFORE this runs) when [`subtree_has_theta_capturing_input`] reports such a
/// capture — that guard follows `(%ref self …)` edges into bindings, so a θ param
/// captured through a derived binding is caught too. The through-binding inline
/// here does not bypass it: it runs only on a density subtree already cleared of
/// θ-capturing reification inputs.
// NOTE: a θ-dependent derived binding both inlined here and captured as a
// `%specinputs` boundary entry leaves a vestigial specinput on the dangling
// binding (density value still correct, `is_flatpdl` passes, but an eager
// evaluator could choke; no fixture triggers this). `infer` rejects reference
// cycles among θ-dependent bindings upstream, so `building` below is defensive.
fn substitute_refs_by_name(m: &mut Module, root: NodeId, map: &[(Symbol, NodeId)]) -> NodeId {
    // Pre-build the θ-inlined copy of every θ-dependent binding referenced as a
    // `(%ref self …)` leaf anywhere in `root`'s syntactic (children) tree — the
    // exact set of self-refs the `map_tree` pass below will meet. Each build
    // recurses into its own θ-dependent dependencies first, so `memo` is fully
    // populated (deps before dependents) by the time the top-level pass reads it.
    // `memo[name] = Some(sub)` marks a θ-dependent binding (inline `sub`);
    // `None` marks a binding that does not depend on θ (leave the ref).
    let mut memo: std::collections::HashMap<Symbol, Option<NodeId>> =
        std::collections::HashMap::new();
    let mut building: std::collections::HashSet<Symbol> = std::collections::HashSet::new();
    for name in syntactic_self_ref_names(m, root) {
        if !map.iter().any(|(n, _)| *n == name) {
            theta_inlined_binding(m, name, map, &mut memo, &mut building);
        }
    }
    crate::driver::map_tree(m, root, &mut |m, id| {
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            // A θ field → its value node (the original per-query substitution).
            if let Some((_, value)) = map.iter().find(|(n, _)| n == name) {
                return Some(*value);
            }
            // A θ-dependent derived binding → its θ-inlined RHS copy.
            if let Some(Some(sub)) = memo.get(name) {
                return Some(*sub);
            }
        }
        None
    })
}

/// The distinct `(%ref self <name>)` names appearing in `root`'s SYNTACTIC subtree
/// — the `children()` walk only, NOT descending into referenced bindings nor into
/// reification `Inputs` — i.e. exactly the self-ref leaves
/// [`crate::driver::map_tree`] will encounter when rewriting `root`. `seen` guards
/// shared DAG nodes.
fn syntactic_self_ref_names(m: &Module, root: NodeId) -> Vec<Symbol> {
    let mut out: Vec<Symbol> = Vec::new();
    let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            if !out.contains(name) {
                out.push(*name);
            }
            continue; // a ref is a leaf — it has no children
        }
        for c in m.node(id).children() {
            stack.push(c);
        }
    }
    out
}

/// Build (memoized) the θ-inlined copy of binding `name`'s RHS, returning
/// `Some(sub)` when `name` is θ-DEPENDENT (its RHS transitively references a θ
/// param, so `sub != rhs`) and `None` when it is not (leave the shared ref).
///
/// θ-dependence is detected structurally: after θ-inlining `name`'s RHS (θ fields
/// → θ values, and nested θ-dependent binding refs → their memoized copies), the
/// node is UNCHANGED iff no θ influence was reachable. So a plain `c = 5` / `f_a =
/// par -> c * par` maps to `None` (untouched), while `a = f_a(theta2)` maps to
/// `Some(f_a(<θ.theta2>))`.
///
/// Only `children()`-reachable self-refs are followed (matching
/// [`crate::driver::map_tree`]); a self-ref sitting solely in a reification
/// `Inputs` boundary is left to the [`subtree_has_theta_capturing_input`] refuse
/// (a θ-capturing boundary already refused upstream). `building` bounds the
/// recursion against reference cycles: a binding met while it is still being built
/// (a malformed non-DAG cycle) yields `None` (left as a ref) rather than looping.
fn theta_inlined_binding(
    m: &mut Module,
    name: Symbol,
    map: &[(Symbol, NodeId)],
    memo: &mut std::collections::HashMap<Symbol, Option<NodeId>>,
    building: &mut std::collections::HashSet<Symbol>,
) -> Option<NodeId> {
    if let Some(done) = memo.get(&name) {
        return *done;
    }
    let Some(bid) = m.binding_by_name(name) else {
        return None; // unbound self-ref: nothing to inline
    };
    let rhs = m.binding(bid).rhs;
    if !building.insert(name) {
        // Reference cycle — terminate rather than recurse. A well-formed FlatPPL
        // module's bindings are a DAG, so this only guards a malformed input.
        return None;
    }
    // Populate `memo` for every θ-dependent binding referenced (as a child leaf)
    // by this RHS, so the `map_tree` pass below can inline them wholesale.
    for r in syntactic_self_ref_names(m, rhs) {
        if !map.iter().any(|(n, _)| *n == r) {
            theta_inlined_binding(m, r, map, memo, building);
        }
    }
    let sub = crate::driver::map_tree(m, rhs, &mut |m, id| {
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: n,
        }) = m.node(id)
        {
            if let Some((_, value)) = map.iter().find(|(nm, _)| nm == n) {
                return Some(*value);
            }
            if let Some(Some(inner)) = memo.get(n) {
                return Some(*inner);
            }
        }
        None
    });
    building.remove(&name);
    // `map_tree` returns the ORIGINAL node id when nothing changed → not
    // θ-dependent; a fresh id → θ-dependent (some θ value / dependent binding was
    // inlined).
    let result = (sub != rhs).then_some(sub);
    memo.insert(name, result);
    result
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

    // A reified-kernel *application* `k(input)` (a `%call(User(k), [input])`)
    // is not a builtin-named op; β-reduce it to its measure body and recurse.
    if let Some(reduced) = crate::kernel::reduce_kernel_application(m, measure_node) {
        return lower_measure_density(m, reduced, v);
    }

    // Dispatch on the measure op.
    let op = builtin_name(m, measure_node);

    match op {
        Some("record") => lower_record_of_draws(m, measure_node, v),
        // A `lawof`-wrapped measure reached as a SUB-measure (e.g. a
        // `bayesupdate` prior — `lower_bayesupdate` hands its prior straight to
        // this dispatcher, unlike the `logdensityof(lawof(M), v)` query ENTRY
        // point, which strips a top-level `lawof` via `measure_of_arg` before
        // ever reaching here). Unwrap to `M` and recurse — idempotent-safe: an
        // already-stripped `lawof` never re-enters this arm, since the entry
        // point's strip means `measure_node` is `M` itself by the time it's
        // dispatched, so there's no double count.
        Some("lawof") => lower_lawof(m, measure_node, v),
        Some("weighted") => lower_weighted(m, measure_node, v),
        Some("logweighted") => lower_logweighted(m, measure_node, v),
        Some("superpose") => lower_superpose(m, measure_node, v),
        Some("normalize") => lower_normalize(m, measure_node, v),
        Some("truncate") => lower_truncate(m, measure_node, v),
        Some("pushfwd") => lower_pushfwd(m, measure_node, v),
        // kchain marginal: discrete-finite latent → mass-weighted logsumexp;
        // continuous / infinite-discrete / non-enumerable → refuse (Task 5).
        Some("kchain") => crate::marginal::lower_kchain_marginal(m, measure_node, v),
        Some("jointchain") => crate::jointchain::lower_jointchain(m, measure_node, v),
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
        // `likelihoodof` / `joint_likelihood` / `bayesupdate` are normally
        // intercepted at the `logdensityof` entry (where the `bundle` needed to
        // lower their likelihood term is in scope; `bayesupdate`'s arg2 and
        // `likelihoodof`'s are θ, not a variate); reaching the measure dispatcher
        // means they were entered as a bare measure — refuse (safety net) rather
        // than emit `builtin_logdensityof(joint_likelihood, …)`. `restrict` is
        // likewise normally intercepted BEFORE the density query, at the driver's
        // `restrict` scan (desugared into `bayesupdate(likelihoodof(kernel, x),
        // marginal)`, §06 "Measure restriction"); reaching here means the
        // desugaring did not fire, so refuse rather than treat `restrict` as a
        // primitive constructor via the fallthrough below.
        // `locscale(m, shift, scale)` = `pushfwd(x -> scale * x + shift, m)`
        // (§06 line 369/402): the affine change-of-variables, reusing the same
        // scalar / matrix-affine synthesis as `pushfwd` (Task 5).
        Some("locscale") => lower_locscale(m, measure_node, v),
        Some("markovchain")
        | Some("kscan")
        | Some("bayesupdate")
        | Some("disintegrate")
        | Some("restrict")
        | Some("likelihoodof")
        | Some("joint_likelihood") => Err(refuse_op(measure_node, m)),
        // Fallthrough: treat as a primitive distribution constructor.
        _ => build_density_term(m, measure_node, v),
    }
}

/// `lawof(M)` reached as a sub-measure (e.g. a `bayesupdate` prior): unwrap to
/// `M` and recurse through [`lower_measure_density`] (spec §06: `lawof` names
/// the law of its argument without changing it — `lawof(draw m) ≡ m`, and a
/// `lawof` over a record of `~`-bound draws is the joint law of those draws, so
/// the unwrapped `M` reaching `lower_record_of_draws` via the `"record"` arm is
/// exactly the shape that arm already scores).
///
/// A `lawof` with anything but exactly one argument refuses (refuse-don't-mislower).
fn lower_lawof(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let arg = {
        let c = expect_builtin_call(m, node, "lawof")
            .ok_or_else(|| refuse(node, m, "expected lawof"))?;
        if c.args.len() != 1 {
            return Err(refuse(node, m, "lawof expects 1 arg"));
        }
        c.args[0]
    };
    lower_measure_density(m, arg, v)
}

// ---------------------------------------------------------------------------
// Record-of-independent-draws (Task 3)
// ---------------------------------------------------------------------------

/// One scored component of an independent product: the component measure node
/// (a `draw`'s argument, e.g. the `Normal(..)` constructor), the pinned variate
/// value node from `v`, and — when the component reached us through a binding
/// reference — that binding, so the driver can pin it to the scored value.
struct Component {
    /// The distribution-constructor (or combinator) node `mᵢ`. For a
    /// bijection-transformed field (`transform = Some(g)`), this is the INNER
    /// draw's measure `Mᵢ`; the driver wraps it as `pushfwd(g, Mᵢ)` before
    /// scoring.
    measure: NodeId,
    /// The matching part of `v` to score `mᵢ` at.
    pinned: NodeId,
    /// `Some(bid)` when the component reached us through a `(%ref self x)`
    /// binding, so the driver can pin that binding to the scored value. For a
    /// transformed field this is the OUTER binding (`sigma` in `sigma =
    /// sqrt(sigma2)`) — pinning it to the scored value feeds sibling measures
    /// and the likelihood that reference `sigma`.
    draw_binding: Option<BindingId>,
    /// `Some(g)` when the field is a unary bijection `g(draw)` (spec §06
    /// pushfwd) rather than a bare `draw`: the field's law is the pushforward
    /// of the inner draw's law under the built-in `g` (e.g. `sqrt`).
    transform: Option<Symbol>,
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

    // Build density terms per component. A bijection-transformed field
    // (`transform = Some(g)`) is scored as the pushforward `pushfwd(g, Mᵢ)` of
    // the inner draw's law under `g` — `lower_pushfwd` applies the §06
    // change-of-variables (`logdensityof(Mᵢ, g⁻¹(y)) − logvol(g⁻¹(y))`),
    // reusing the recorded/derived inverse; a non-invertible `g` refuses there.
    let mut terms: Vec<NodeId> = Vec::with_capacity(components.len());
    for comp in &components {
        let measure = match comp.transform {
            None => comp.measure,
            Some(g) => {
                let fwd = m.alloc(Node::Const(g));
                build_call(m, "pushfwd", &[fwd, comp.measure])
            }
        };
        terms.push(lower_measure_density(m, measure, comp.pinned)?);
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

    // The scored value may be a NAMED binding referring to a record literal
    // (`theta = record(...)`, i.e. a `Ref(SelfMod, theta)`), not an inline
    // `record(...)`. Resolve one ref level — as the measure side does in
    // `lower_measure_density` — so a ref-to-record variate destructures the same
    // as the inline form. A deeper ref-to-ref chain still refuses (one level,
    // matching the measure side).
    let (v, _) = resolve_ref_one(m, v);
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

        let (measure, draw_binding, transform) = resolve_component_draw(m, field.value)
            .ok_or_else(|| {
                refuse(
                    field.value,
                    m,
                    "field is not a draw, a reference to a draw, or a bijection of a draw",
                )
            })?;
        components.push(Component {
            measure,
            pinned,
            draw_binding,
            transform,
        });
    }

    Ok(components)
}

/// Resolve a record-field value to the measure to score it at. Returns
/// `(measure_node, outer_binding, transform)`:
/// * `measure_node` — the inner draw's measure argument `Mᵢ` (the
///   distribution-constructor node);
/// * `outer_binding` — `Some(bid)` when the field reached us through a
///   `(%ref self x)` binding, so the driver can pin `x` to the scored value;
/// * `transform` — `Some(g)` when the field is a unary bijection `g(draw)`
///   (§06 pushfwd) rather than a bare draw; the driver wraps `Mᵢ` as
///   `pushfwd(g, Mᵢ)` before scoring.
///
/// Cases: **A** `(%ref self x)` whose binding RHS is `draw(Mᵢ)`; **B** inline
/// `draw(Mᵢ)`; **C** a unary builtin call `g(inner)` (either inline or the RHS
/// of a `(%ref self x)` binding) where `inner` resolves — one further ref hop —
/// to a `draw(Mᵢ)`. `sigma = sqrt(sigma2)` is Case C: `outer_binding` is
/// `sigma`'s binding and `transform = sqrt`.
fn resolve_component_draw(
    m: &Module,
    value: NodeId,
) -> Option<(NodeId, Option<BindingId>, Option<Symbol>)> {
    // One `(%ref self x)` hop: the field either IS a self-ref to a binding
    // (Cases A / ref-C) or is spelled inline (Cases B / inline-C).
    let (effective, outer_binding) = match m.node(value) {
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) => {
            let bid = m.binding_by_name(*name)?;
            (m.binding(bid).rhs, Some(bid))
        }
        _ => (value, None),
    };

    // Cases A / B: the effective RHS is a bare `draw(Mᵢ)`.
    if let Some(measure) = draw_argument(m, effective) {
        return Some((measure, outer_binding, None));
    }

    // Case C: the effective RHS is a unary built-in call `g(inner)` where
    // `inner` resolves (one more ref hop) to a `draw(Mᵢ)`. The field's law is
    // the pushforward of the inner draw's law under `g`. `g` need not be a
    // recognised bijection here — the driver's `pushfwd(g, Mᵢ)` lowering
    // refuses a non-invertible `g` (refuse-don't-mislower).
    if let Node::Call(c) = m.node(effective) {
        if let CallHead::Builtin(g) = c.head {
            if c.args.len() == 1 && c.named.is_empty() {
                if let Some(inner_measure) = resolve_inner_draw_measure(m, c.args[0]) {
                    return Some((inner_measure, outer_binding, Some(g)));
                }
            }
        }
    }
    None
}

/// The inner half of Case C: resolve `node` — one `(%ref self x)` hop or inline
/// — to the measure argument of a `draw(Mᵢ)`. Returns just the measure; the
/// inner draw's binding is NOT pinned (in the transformed-field models it is
/// referenced only to define the outer binding, which the driver pins instead).
fn resolve_inner_draw_measure(m: &Module, node: NodeId) -> Option<NodeId> {
    let effective = match m.node(node) {
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) => m.binding(m.binding_by_name(*name)?).rhs,
        _ => node,
    };
    draw_argument(m, effective)
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

    // Closed-form Z for a CONVEX SUPERPOSITION OF NORMALIZED MIXANDS:
    // `normalize(superpose(weighted(w₁, A₁), …, weighted(wₖ, Aₖ)))` where every
    // mixand Aᵢ is a probability measure and every weight wᵢ is a
    // variate-independent scalar. Total mass is additive over the superposition
    // and multiplicative under `weighted` (§06 "Additive superposition",
    // "Density reweighting"):
    //   Z = totalmass(superpose(weighted(wᵢ, Aᵢ)))
    //     = Σ wᵢ · totalmass(Aᵢ) = Σ wᵢ           (Aᵢ normalized ⇒ totalmass = 1).
    // So Z = Σ wᵢ is a closed-form deterministic scalar, and (reusing the
    // existing `superpose`/`weighted` density lowering for the numerator)
    //   logdensityof(normalize(sup), v)
    //     = logdensityof(sup, v) − log(Σ wᵢ)
    //     = logsumexp([log wᵢ + logdensityof(Aᵢ, v)]) − log(Σ wᵢ).
    // The weights-sum-to-one mixture idiom the spec names as canonical (§06
    // "Additive superposition": `normalize(superpose(weighted(w1, M1),
    // weighted(w2, M2)))`) is the Z = 1 special case — `log(Σ wᵢ) = log 1 = 0` —
    // handled by the SAME rule, with no symbolic sum-to-one proof. A
    // variate-DEPENDENT weight makes `totalmass(weighted(w, A)) = ∫ w(x) dA(x)` a
    // v-dependent integral with no scalar closed form, so such a superposition
    // falls through to the refuse below (refuse-don't-mislower). A degenerate
    // Z (all weights zero → `log(0) = −∞`; an infinite weight → `log(inf) = ∞`)
    // violates §06's Z finite & nonzero precondition and is the backend's runtime
    // concern, consistent with the truncate and Gaussian-product arms above.
    if let Some(weights) = recognize_convex_superposition(m, m_inner_resolved) {
        let sup_density = lower_measure_density(m, m_inner, v)?;
        let z = fold_add(m, &weights);
        let log_z = build_call(m, "log", &[z]);
        return Ok(build_call(m, "sub", &[sup_density, log_z]));
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

/// Recognize `superpose(weighted(w₁, A₁), …, weighted(wₖ, Aₖ))` (k ≥ 2) as a
/// convex superposition of normalized mixands: every component is a `weighted`
/// whose base measure Aᵢ is a probability measure (`Mass::Normalized`) and whose
/// weight wᵢ is a variate-INDEPENDENT scalar. Returns the weight nodes
/// (w₁ … wₖ) in order — their sum is the closed-form normalizer Z = Σ wᵢ (see
/// [`lower_normalize`]) — or `None` if any component does not match, in which
/// case `normalize` keeps its refuse (refuse-don't-mislower).
///
/// Each mixand's normalized-ness is read from its INFERRED type: a leaf §08
/// distribution, a `lawof`, a `Dirac`, or a normalized combinator all carry
/// `Mass::Normalized` (`fill_mass`). A bare (unweighted) component, a mixand that
/// is not statically normalized, or a variate-dependent weight all yield `None` —
/// the recognizer never assumes a mass it cannot read off the type. Structural,
/// immutable read of `m`.
fn recognize_convex_superposition(m: &Module, node: NodeId) -> Option<Vec<NodeId>> {
    let components: Vec<NodeId> = {
        let sup = expect_builtin_call(m, node, "superpose")?;
        if sup.args.len() < 2 {
            return None;
        }
        sup.args.to_vec()
    };
    let mut weights = Vec::with_capacity(components.len());
    for comp in components {
        let (comp_resolved, _) = resolve_ref_one(m, comp);
        let (weight, base) = {
            let w = expect_builtin_call(m, comp_resolved, "weighted")?;
            if w.args.len() != 2 {
                return None;
            }
            (w.args[0], w.args[1])
        };
        // The mixand must be a probability measure — its Z contribution is then
        // wᵢ · totalmass(Aᵢ) = wᵢ. Read the mass off the inferred type; anything
        // not statically `Mass::Normalized` (Finite/LocallyFinite/Unknown) has no
        // closed-form unit mass, so bail.
        let (base_resolved, _) = resolve_ref_one(m, base);
        let base_normalized = matches!(
            m.type_of(base_resolved),
            Some(Type::Measure {
                mass: Mass::Normalized,
                ..
            })
        );
        if !base_normalized {
            return None;
        }
        // The weight must be variate-independent: a v-dependent weight makes
        // `totalmass(weighted(w, A)) = ∫ w(x) dA(x)` a v-dependent integral, not
        // the scalar Σ wᵢ closed form.
        if weight_is_variate_dependent(m, weight) {
            return None;
        }
        weights.push(weight);
    }
    Some(weights)
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
/// refuse-don't-mislower). Structural matching; takes `&mut` only so the shared
/// [`normal_ctor`]/[`split_kernel_constructor`] can intern positional-arg names.
fn recognize_gaussian_product(m: &mut Module, logweighted: NodeId) -> Option<GaussianProduct> {
    let (lw_node, base) = {
        let c = expect_builtin_call(m, logweighted, "logweighted")?;
        if c.args.len() != 2 {
            return None;
        }
        (c.args[0], c.args[1])
    };

    // Factor 1 (g1): the logweighted base is a `Normal` constructor.
    let (g1, g1_kwargs) = normal_ctor(m, base)?;
    let mu1 = find_kwarg(m, &g1_kwargs, "mu")?;
    let sigma1 = find_kwarg(m, &g1_kwargs, "sigma")?;

    // Factor 2 (g2): ℓ is a reified function `x → logdensityof(g2, x)` — a
    // `functionof` whose single positional body scores a `Normal` AT the reified
    // argument `x` (not at a constant, which would make ℓ a constant weight, not a
    // second Gaussian factor).
    let (lw_resolved, _) = resolve_ref_one(m, lw_node);
    // Read the reified callable's single body and input placeholder (`x`), then
    // drop the `&Call` borrow before `gaussian_factor_scored_at` reborrows `&mut`.
    let (g2_body, arg_ref) = {
        let f = expect_builtin_call(m, lw_resolved, "functionof")?;
        if f.args.len() != 1 {
            return None;
        }
        // g2 must be scored at exactly this ref, so ℓ(x) = logdensityof(g2, x).
        let arg_ref = match &f.inputs {
            Some(Inputs::Spec(entries)) if entries.len() == 1 => entries[0].1,
            _ => return None,
        };
        (f.args[0], arg_ref)
    };
    let (mu2, sigma2) = gaussian_factor_scored_at(m, g2_body, arg_ref)?;

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
fn gaussian_factor_scored_at(
    m: &mut Module,
    body: NodeId,
    arg_ref: Ref,
) -> Option<(NodeId, NodeId)> {
    // Not-yet-lowered: logdensityof(Normal(...), arg). Read the scored ctor node
    // out of the `&Call` (and check it is scored at `arg_ref`) before handing it
    // to the `&mut`-borrowing `normal_ctor`.
    let g2_ctor = {
        match expect_builtin_call(m, body, "logdensityof") {
            Some(ld) if ld.args.len() == 2 && is_ref_to(m, ld.args[1], arg_ref) => Some(ld.args[0]),
            Some(_) => return None,
            None => None,
        }
    };
    if let Some(g2_ctor) = g2_ctor {
        let (_g2, kwargs) = normal_ctor(m, g2_ctor)?;
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

/// If `node` (after one ref hop) is a `Normal(...)` constructor (positional or
/// keyword arguments), return `(constructor_node, args)` with each argument
/// bound to its parameter name; otherwise `None`.
fn normal_ctor(m: &mut Module, node: NodeId) -> Option<(NodeId, Vec<(Symbol, NodeId)>)> {
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
/// **Positional-first-class.** Delegates to [`split_kernel_constructor`] — the
/// same helper [`build_density_term`] and `sample::split_constructor` use — so
/// a positionally-written base (`Normal(0.0, 1.0)`), a keyword base, or a mixed
/// form all read to the identical `(ctor_sym, kwargs)` pair (spec §04 calling
/// conventions: positional args bind to the constructor's ordered §08
/// parameter names). `None` (not a builtin constructor call, or a malformed
/// positional/keyword mix) refuses cleanly rather than mislowering.
fn kernel_and_input(m: &mut Module, ctor: NodeId) -> Result<(NodeId, NodeId), RefuseError> {
    let (ctor_resolved, _) = resolve_ref_one(m, ctor);
    let (ctor_sym, kwargs) = split_kernel_constructor(m, ctor_resolved).ok_or_else(|| {
        refuse(
            ctor_resolved,
            m,
            "truncation base must be a built-in constructor call with well-formed positional or keyword arguments",
        )
    })?;
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
/// The forward map may be given explicitly as a `bijection(f, f_inv, logvol)`
/// node (directly or via one level of ref), OR — per §06 case 1 — as a known
/// invertible builtin (`pushfwd(exp, M)`, `pushfwd(x -> exp(x), M)`), for which
/// `(f_inv, logvol)` is synthesised analytically by [`crate::invert`]. Anything
/// else refuses.
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

    // Extract f_inv and logvol: from an explicit bijection node if present,
    // otherwise synthesise them for a known invertible forward builtin.
    // Structural projection (§06 case 2): `pushfwd(fn(get(_, [fields])), M)` is a
    // MARGINALIZATION, not a bijection — `get` has no inverse, so it never reaches
    // the change-of-variables path. Recognise it up front and lower the closed-form
    // marginal over the SELECTED components (the unselected components integrate to
    // 1 and drop). Only over an explicit field-keyed product; otherwise refuse.
    if let Some(fields) = recognize_get_projection(m, bij_resolved) {
        return lower_projection_pushfwd(m, m_inner, &fields, v);
    }

    let (f_inv_node, logvol_node) =
        if let Some(bij) = expect_builtin_call(m, bij_resolved, "bijection") {
            if bij.args.len() != 3 {
                return Err(refuse(
                    bij_resolved,
                    m,
                    "bijection expects 3 args (f, f_inv, logvol)",
                ));
            }
            (bij.args[1], bij.args[2])
        } else {
            // Not an explicit bijection: try analytic synthesis (§06 case 1). Pass
            // `M`'s variate domain (needed for domain-restricted maps like `pow`);
            // an unknown domain defaults to `%any`, which those maps refuse.
            let domain = match m.type_of(m_inner) {
                Some(Type::Measure { domain, .. }) => (**domain).clone(),
                _ => Type::Any,
            };
            // Also thread `M`'s refined SUPPORT (`valueset_of`, e.g. `posreals`
            // for `Gamma`, `nonnegreals` for `Exponential`): the coarse variate
            // type is `scalar real` (natural extent `reals`), which would refuse
            // every positive-support base for `log`/`pow`. `None`/`%unknown`
            // support falls back to `Unknown` — conservatively refused by the
            // positivity guard (refuse-don't-mislower), NOT defaulted to positive.
            let support = m
                .valueset_of(m_inner)
                .cloned()
                .unwrap_or(flatppl_core::ValueSet::Unknown);
            match crate::invert::derive_bijection(m, bij_node, &domain, &support)? {
                Some(bij) => (bij.f_inv, bij.logvol),
                None => {
                    return Err(refuse(
                        bij_resolved,
                        m,
                        "pushfwd bijection arg must be a bijection(f, f_inv, logvol) node",
                    ));
                }
            }
        };

    // preimage = f_inv(v)
    let preimage = build_user_call(m, f_inv_node, v);
    // inner_density = logdensityof(M, preimage)
    let inner_density = lower_measure_density(m, m_inner, preimage)?;
    // logvol_val = logvol(preimage)
    let logvol_val = build_user_call(m, logvol_node, preimage);
    Ok(build_call(m, "sub", &[inner_density, logvol_val]))
}

/// `logdensityof(locscale(m, shift, scale), v)` — the affine (location-scale)
/// pushforward `pushfwd(x -> scale * x + shift, m)` (§06 line 369/402). We derive
/// the change-of-variables `(f_inv, logvol)` from the affine parameters directly
/// ([`crate::invert::derive_locscale`], which reuses the same scalar / matrix-
/// affine emission as [`lower_pushfwd`]'s synthesis path) and apply the §06
/// change-of-variables formula `logdensityof(m, f_inv(v)) − logvol(f_inv(v))` —
/// structurally identical to [`lower_pushfwd`]'s tail.
fn lower_locscale(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (m_inner, shift, scale) = {
        let c = expect_builtin_call(m, node, "locscale")
            .ok_or_else(|| refuse(node, m, "expected locscale"))?;
        if c.args.len() != 3 {
            return Err(refuse(
                node,
                m,
                "locscale expects 3 positional args (m, shift, scale)",
            ));
        }
        (c.args[0], c.args[1], c.args[2])
    };
    // The affine form is keyed on `m`'s variate domain (scalar vs vector); read it
    // from the inner measure's inferred type (an unknown domain → `%any`, which
    // `derive_locscale` refuses).
    let domain = match m.type_of(m_inner) {
        Some(Type::Measure { domain, .. }) => (**domain).clone(),
        _ => Type::Any,
    };
    let bij = crate::invert::derive_locscale(m, shift, scale, &domain)?;
    // §06 change-of-variables: logdensityof(m, f_inv(v)) − logvol(f_inv(v)).
    let preimage = build_user_call(m, bij.f_inv, v);
    let inner_density = lower_measure_density(m, m_inner, preimage)?;
    let logvol_val = build_user_call(m, bij.logvol, preimage);
    Ok(build_call(m, "sub", &[inner_density, logvol_val]))
}

/// Recognise a **pure structural-projection** forward function
/// `fn(get(_, ["a", "c", …]))` — a one-input `functionof` lambda whose body is
/// exactly `get(<the input placeholder>, vector("a", "c", …))` (§06 case 2). On
/// a match, return the selected field-name strings, in selection order.
///
/// The projection must be PURE: the `get`'s first argument is the bare input
/// placeholder (no wrapping transform) and its second argument is a `vector`
/// literal of field-name strings. A `get` that also transforms, indexes by
/// position, or selects via a computed key is NOT this pattern (`None`) — the
/// caller then treats the forward as a bijection candidate / refuses.
///
/// Field names are returned as owned strings (not interned `Symbol`s) so this
/// stays an immutable read; the caller matches them against the product's
/// component names by resolving each component symbol to its string.
fn recognize_get_projection(m: &Module, f: NodeId) -> Option<Vec<Box<str>>> {
    // f = functionof with exactly one %local placeholder input.
    let Node::Call(c) = m.node(f) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "functionof" || c.args.len() != 1 {
        return None;
    }
    let Some(Inputs::Spec(entries)) = &c.inputs else {
        return None;
    };
    if entries.len() != 1 || entries[0].1.ns != RefNs::Local {
        return None;
    }
    let ph = entries[0].1.name;

    // body = get(<ph>, vector("a", …)) — a pure field selection.
    let get = expect_builtin_call(m, c.args[0], "get")?;
    if get.args.len() != 2 || !get.named.is_empty() {
        return None;
    }
    // First arg is EXACTLY the input placeholder (no transform).
    if !matches!(m.node(get.args[0]), Node::Ref(Ref { ns: RefNs::Local, name }) if *name == ph) {
        return None;
    }
    // Second arg is a non-empty `vector` of string literals (the selected fields).
    string_literal_vector(m, get.args[1])
}

/// Read a `vector("a", "b", …)` node as owned STRING-literal strings, in order,
/// or `None` if it is not a non-empty positional vector of string literals. Shared
/// by [`recognize_get_projection`] (the projected FIELD names) and the
/// `relabel(_, [labels])` arm of [`lower_projection_pushfwd`] (the component
/// LABELS) — both read the identical `[ "a", "b", … ]` surface form.
fn string_literal_vector(m: &Module, node: NodeId) -> Option<Vec<Box<str>>> {
    let vec = expect_builtin_call(m, node, "vector")?;
    if !vec.named.is_empty() || vec.args.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(vec.args.len());
    for &a in vec.args.iter() {
        let Node::Lit(Scalar::Str(s)) = m.node(a) else {
            return None; // a non-string entry is not a field/label list
        };
        out.push(s.clone());
    }
    Some(out)
}

/// Lower `pushfwd(fn(get(_, [fields])), M)` — the §06 case-2 structural
/// projection — as the closed-form MARGINAL over the SELECTED components of an
/// explicit field-keyed product `M`. The marginal density is the density of the
/// SUB-PRODUCT over just the selected fields: for an independent product the
/// unselected components integrate to 1, so they drop cleanly and the marginal
/// is the sum of the kept components' densities at the projected point's matching
/// fields (§06 "joint and iid (independent products)").
///
/// We realise this by building the field-keyed SUB-PRODUCT (a keyword `joint`
/// or a record-of-draws over the selected fields) and re-dispatching it through
/// [`lower_measure_density`] against the projected point `v` (a record over the
/// selected fields) — reusing the exact keyword-joint / record-of-draws
/// machinery ([`lower_keyword_joint`] / [`lower_record_of_draws`]) rather than
/// re-deriving the marginal.
///
/// **Index-keyed products via `relabel`.** An `iid(M, n)` or a positional
/// `joint(M₁, …, Mₖ)` carries no field labels, but `relabel(product, [labels])`
/// (§06) names its component slots. The `relabel` arm materialises that named
/// product as a keyword `joint(label₀ = M₀, …)` and re-dispatches through THIS
/// function's keyword-joint arm, so the labels supply the field names and the
/// selected-name projection, the mass-preservation drop guard, and the sub-joint
/// density all reuse the field-keyed path (the §06 canonical example
/// `pushfwd(fn(get(_, ["a","c"])), relabel(iid(Normal(0,1),3), ["a","b","c"]))`).
///
/// **Scope (refuse-don't-mislower).** `jointchain` is a DEPENDENT product — a
/// component's kernel reads earlier variates, so marginalizing one is a `kchain`
/// integral, not the free drop the independent-product identity permits; it
/// REFUSES (naming jointchain), both directly and through `relabel`. A bare
/// `iid` / positional `joint` projected by field NAME (no `relabel` to name its
/// slots), or a NON-product measure, likewise has no closed-form field-keyed
/// marginal here and refuses (§06 case 2 permits "compute numerically or report a
/// static error").
fn lower_projection_pushfwd(
    m: &mut Module,
    m_inner_expr: NodeId,
    fields: &[Box<str>],
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    let (m_inner, _) = resolve_ref_one(m, m_inner_expr);
    match builtin_name(m, m_inner) {
        // Keyword `joint(a = Mₐ, …)`: the selected components form a sub-joint.
        Some("joint") => {
            let named = {
                let c = expect_builtin_call(m, m_inner, "joint")
                    .ok_or_else(|| refuse(m_inner, m, "expected joint"))?;
                if !c.args.is_empty() {
                    return Err(refuse(
                        m_inner,
                        m,
                        "structural projection needs a FIELD-KEYED product; a positional joint has \
                         no field labels to select — refuse rather than mislower",
                    ));
                }
                c.named.to_vec()
            };
            // A joint field's value IS its component measure directly.
            refuse_unnormalized_dropped_fields(m, m_inner, &named, fields, |_m, value| {
                Some(value)
            })?;
            let selected = select_projection_fields(m, m_inner, &named, fields)?;
            let joint_sym = m.intern("joint");
            let sub = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(joint_sym),
                args: Vec::<NodeId>::new().into(),
                named: selected.into(),
                inputs: None,
            }));
            lower_measure_density(m, sub, v)
        }
        // Record-of-draws `record(a = draw(Mₐ), …)`: the selected draws form a
        // sub-record.
        Some("record") => {
            let named = {
                let c = expect_builtin_call(m, m_inner, "record")
                    .ok_or_else(|| refuse(m_inner, m, "expected record"))?;
                if !c.args.is_empty() {
                    return Err(refuse(
                        m_inner,
                        m,
                        "structural projection needs a field-keyed record; positional args present \
                         — refuse rather than mislower",
                    ));
                }
                c.named.to_vec()
            };
            // A record field's value is `draw(Mₐ)` (or a ref to one) — unwrap to
            // the underlying measure argument before reading its mass.
            refuse_unnormalized_dropped_fields(m, m_inner, &named, fields, |m, value| {
                resolve_component_draw(m, value).map(|(measure, _, _)| measure)
            })?;
            let selected = select_projection_fields(m, m_inner, &named, fields)?;
            let record_sym = m.intern("record");
            let sub = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(record_sym),
                args: Vec::<NodeId>::new().into(),
                named: selected.into(),
                inputs: None,
            }));
            lower_measure_density(m, sub, v)
        }
        // `relabel(product, [labels])`: the labels name an index-keyed product's
        // component slots (§06). Materialise the named product as a keyword
        // `joint(label₀ = M₀, …)` and re-dispatch through the keyword-joint arm
        // above — reusing the field-keyed projection, the drop guard, and the
        // sub-joint density rather than re-deriving an index-remapped marginal.
        Some("relabel") => {
            let (inner, labels_node) = {
                let c = expect_builtin_call(m, m_inner, "relabel")
                    .ok_or_else(|| refuse(m_inner, m, "expected relabel"))?;
                if c.args.len() != 2 {
                    return Err(refuse(
                        m_inner,
                        m,
                        "relabel expects 2 args (measure, labels)",
                    ));
                }
                (c.args[0], c.args[1])
            };
            let labels = string_literal_vector(m, labels_node).ok_or_else(|| {
                refuse(
                    m_inner,
                    m,
                    "relabel labels must be a vector of string literals to name the \
                     projected product's component slots",
                )
            })?;
            // `relabel(jointchain, [labels])`: a jointchain is a DEPENDENT product
            // whose variates are ALREADY field-keyed (the base variate's field,
            // then each kernel's field). A relabel here does not rename the chain's
            // record domain — inference leaves it unchanged — so labels that DIFFER
            // from the chain's own variate fields are an ill-formed rename with no
            // well-defined point mapping (refuse). An IDENTITY relabel (labels equal
            // the chain's variate fields, in order) is a no-op wrapper: strip it and
            // re-dispatch to the bare `jointchain` prefix-keep arm below — the
            // projected names and the point are already keyed by the internal
            // field names.
            let (inner_resolved, _) = resolve_ref_one(m, inner);
            if builtin_name(m, inner_resolved) == Some("jointchain") {
                let chain_fields = crate::jointchain::record_variate_fields(m, inner_resolved)
                    .ok_or_else(|| refuse_jointchain_projection(m, inner_resolved))?;
                let identity = chain_fields.len() == labels.len()
                    && chain_fields
                        .iter()
                        .zip(labels.iter())
                        .all(|(sym, label)| m.resolve(*sym) == label.as_ref());
                if !identity {
                    return Err(refuse_jointchain_relabel_rename(m, inner_resolved));
                }
                return lower_projection_pushfwd(m, inner, fields, v);
            }
            let components = independent_product_components(m, inner)?;
            if labels.len() != components.len() {
                return Err(refuse(
                    m_inner,
                    m,
                    &format!(
                        "relabel gives {} label(s) for a {}-component product — a relabel \
                         names each component slot exactly once — refuse rather than mislower",
                        labels.len(),
                        components.len()
                    ),
                ));
            }
            let mut named = Vec::with_capacity(labels.len());
            for (label, &comp) in labels.iter().zip(components.iter()) {
                let name = m.intern(label.as_ref());
                named.push(NamedArg {
                    kind: NamedKind::Field,
                    name,
                    value: comp,
                });
            }
            let joint_sym = m.intern("joint");
            let materialized = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(joint_sym),
                args: Vec::<NodeId>::new().into(),
                named: named.into(),
                inputs: None,
            }));
            lower_projection_pushfwd(m, materialized, fields, v)
        }
        // `jointchain` is a DEPENDENT product: a component's kernel reads earlier
        // variates. A projection keeping a dependency-respecting LEADING PREFIX of
        // the chain is closed-form (the dropped trailing kernels are normalized
        // Markov kernels that integrate to 1 and drop, so the marginal is the
        // sub-jointchain over the kept prefix); ANY other keep drops a
        // depended-upon variate — the intractable `kchain` integral — and refuses.
        // Reached for a bare jointchain projected by name and for the identity
        // `relabel(jointchain, …)` re-dispatch above.
        Some("jointchain") => lower_jointchain_projection(m, m_inner, fields, v),
        other => Err(refuse(
            m_inner,
            m,
            &format!(
                "structural projection (§06 case 2) is closed-form only over an explicit \
                 FIELD-KEYED product (keyword `joint` / record-of-draws) or an index-keyed \
                 product named by `relabel` (iid / positional joint); got `{}` — a bare \
                 index-keyed product projected by field name, or a non-product measure, has \
                 no closed-form field-keyed marginal here — refuse rather than mislower",
                other.unwrap_or("<non-builtin>")
            ),
        )),
    }
}

/// The refusal for a structural projection over a `jointchain` whose shape is not
/// a record-family single-draw chain (scalar-cat family, keyword-form, or
/// malformed) — there are no field-keyed variates to project by name.
fn refuse_jointchain_projection(m: &Module, node: NodeId) -> RefuseError {
    refuse(
        node,
        m,
        "structural projection over a `jointchain` is unsupported here: jointchain is a \
         DEPENDENT product (a component's kernel reads earlier variates), and this chain has \
         no field-keyed variates to select by name (scalar-cat / keyword-form / malformed) — \
         marginalizing a component is otherwise a `kchain` integral — refuse rather than \
         mislower",
    )
}

/// The refusal for a NON-PREFIX keep over a `jointchain`: the kept field set is
/// not a leading prefix, so it drops a leading or interior variate that a kept or
/// later kernel depends on — the intractable `kchain` integral, not the free drop
/// a dependency-respecting prefix keep permits.
fn refuse_jointchain_nonprefix(m: &Module, node: NodeId) -> RefuseError {
    refuse(
        node,
        m,
        "structural projection over a `jointchain` keeps a NON-PREFIX field set: it drops a \
         leading or interior variate that a kept or later kernel depends on, so marginalizing \
         it is the intractable `kchain` integral, not the free drop a dependency-respecting \
         prefix keep permits — only a leading prefix of the chain's variates is closed-form — \
         refuse rather than mislower",
    )
}

/// The refusal for `relabel(jointchain, [labels])` whose labels DIFFER from the
/// chain's own variate fields — an ill-formed rename (inference leaves the
/// jointchain's record domain unchanged, so the relabeled names have no
/// well-defined point mapping onto the dependent chain).
fn refuse_jointchain_relabel_rename(m: &Module, node: NodeId) -> RefuseError {
    refuse(
        node,
        m,
        "structural projection over a `relabel(jointchain, …)` whose labels differ from the \
         chain's own variate fields is an ill-formed rename: inference keeps the jointchain's \
         record domain unchanged, so the relabeled names have no well-defined mapping onto the \
         dependent chain — only an identity relabel (or a bare jointchain) exposes a \
         dependency-respecting prefix keep; marginalizing a depended-upon variate is otherwise \
         the `kchain` integral — refuse rather than mislower",
    )
}

/// Lower a structural projection over a record-family `jointchain` `node` that
/// keeps a **dependency-respecting leading prefix** of the chain's variates
/// (§06 "Density of composed measures").
///
/// jointchain kernels read only PRIOR variates, so keeping the first `k` variate
/// fields is dependency-closed: the dropped trailing kernels are normalized
/// Markov kernels that integrate to 1 and drop, and the marginal is exactly the
/// SUB-jointchain over the kept prefix. We recover the chain's ordered variate
/// fields ([`crate::jointchain::record_variate_fields`]), confirm the kept set is
/// the leading `{0, …, k-1}` (else it drops a depended-upon variate — the
/// intractable `kchain` integral — and refuses), verify each dropped trailing
/// kernel's BODY is a confirmed normalized probability measure
/// ([`crate::jointchain::kernel_body_is_normalized`] — reading the body measure's
/// OWN mass, NOT the kernel-type mass, which inference types as `Normalized`
/// unconditionally; a non-normalized dropped body would silently omit mass,
/// mirroring [`refuse_unnormalized_dropped_fields`] for the independent-product
/// arms), then re-dispatch: the base measure alone for a length-1 prefix, else the
/// sub-`jointchain(base, K₁, …, K_{k-1})` through the existing jointchain density
/// lowering.
fn lower_jointchain_projection(
    m: &mut Module,
    node: NodeId,
    fields: &[Box<str>],
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    // The chain's ordered variate fields (base field, then each kernel's field).
    // `None` ⇒ not a record-family single-draw chain — no field-keyed projection.
    let chain_fields = crate::jointchain::record_variate_fields(m, node)
        .ok_or_else(|| refuse_jointchain_projection(m, node))?;
    let chain_names: Vec<&str> = chain_fields.iter().map(|s| m.resolve(*s)).collect();

    // Map each kept field to its position in the chain; a kept name that is not a
    // chain variate is a mis-projection — refuse.
    let mut kept_idx = Vec::with_capacity(fields.len());
    for f in fields {
        let idx = chain_names
            .iter()
            .position(|n| *n == f.as_ref())
            .ok_or_else(|| {
                refuse(
                    node,
                    m,
                    &format!(
                        "structural projection keeps field `{f}` which is not a variate of this \
                         jointchain — refuse rather than mislower"
                    ),
                )
            })?;
        kept_idx.push(idx);
    }
    let k = kept_idx.len();
    // Dependency-respecting PREFIX keep ⇔ the kept indices are exactly the leading
    // {0, …, k-1}. Anything else (a dropped leading/interior variate) is the
    // intractable kchain integral.
    kept_idx.sort_unstable();
    if !kept_idx.iter().copied().eq(0..k) {
        return Err(refuse_jointchain_nonprefix(m, node));
    }

    let args: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "jointchain")
            .ok_or_else(|| refuse_jointchain_projection(m, node))?;
        c.args.to_vec()
    };
    // The dropped components are the trailing kernels `args[k..]`; each must be a
    // confirmed normalized Markov kernel (∫ K = 1) to drop cleanly. We must NOT
    // read the kernel-TYPE mass here: inference types EVERY `kernelof(...)` as
    // `Mass::Normalized` regardless of body (crates/infer/src/ops.rs), so a
    // trailing kernel whose BODY is an improper (infinite-mass) measure —
    // `Lebesgue`/`Counting` or an un-normalized combinator — would falsely pass a
    // kernel-type mass check and be dropped as if it integrated to 1, silently
    // lowering to a finite WRONG density. Verify the kernel BODY's own output
    // measure is a confirmed normalized probability measure instead.
    for &dropped in &args[k..] {
        if !crate::jointchain::kernel_body_is_normalized(m, dropped) {
            return Err(refuse(
                node,
                m,
                "structural projection over a `jointchain` drops a trailing kernel whose body is \
                 not a confirmed normalized probability measure (its output measure's own mass is \
                 not `Normalized` — e.g. a base/reference measure `Lebesgue`/`Counting`, or an \
                 un-normalized combinator, whose total mass is infinite or unknown); the \
                 kernel-TYPE mass is unreliable here (inference types every `kernelof` as \
                 normalized regardless of body), and dropping such a kernel would silently omit \
                 mass from the marginal — refuse rather than mislower",
            ));
        }
    }

    if k == 1 {
        // Keep only the base variate: the marginal is the base measure's density.
        lower_measure_density(m, args[0], v)
    } else {
        // Keep a ≥2-length prefix: the marginal is the sub-jointchain over the
        // first `k` components (base + first `k-1` kernels). Re-dispatch through
        // the existing jointchain density lowering rather than re-deriving it.
        let jointchain_sym = m.intern("jointchain");
        let sub = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(jointchain_sym),
            args: args[..k].to_vec().into(),
            named: Vec::<NamedArg>::new().into(),
            inputs: None,
        }));
        crate::jointchain::lower_jointchain(m, sub, v)
    }
}

/// Extract the ordered component measures of an INDEPENDENT index-keyed product
/// `inner` under a `relabel(inner, [labels])` projection: `iid(M, n)` yields `n`
/// copies of the SAME `M` node (its independent repeats); a positional
/// `joint(M₁, …, Mₖ)` yields its positional args in order. These are the ONLY two
/// forms whose components are independent AND positionally addressable, so a
/// relabel can name each slot and the unselected slots drop cleanly.
///
/// **Refuses (refuse-don't-mislower):**
/// * `jointchain` — a DEPENDENT product; marginalizing a component is a `kchain`
///   integral (see [`refuse_jointchain_projection`]).
/// * a KEYWORD `joint` — already field-keyed; project it by name directly rather
///   than relabeling.
/// * an `iid` whose size is not a statically-resolved 1-D count — a
///   dynamic/multi-axis product's components can't be named slot-by-slot here.
/// * any other measure — not an index-keyed product with nameable slots.
fn independent_product_components(m: &Module, inner: NodeId) -> Result<Vec<NodeId>, RefuseError> {
    let (inner_resolved, _) = resolve_ref_one(m, inner);
    match builtin_name(m, inner_resolved) {
        Some("iid") => {
            let n = iid_static_size(m, inner_resolved).ok_or_else(|| {
                refuse(
                    inner_resolved,
                    m,
                    "relabel over an iid whose size is not a statically-resolved 1-D count \
                     (dynamic / multi-axis / unresolved) — its components cannot be named \
                     slot-by-slot — refuse rather than mislower",
                )
            })?;
            let mm = {
                let c = expect_builtin_call(m, inner_resolved, "iid")
                    .ok_or_else(|| refuse(inner_resolved, m, "expected iid"))?;
                if c.args.len() != 2 {
                    return Err(refuse(
                        inner_resolved,
                        m,
                        "iid expects 2 args (measure, size)",
                    ));
                }
                c.args[0]
            };
            // `n` copies of the SAME measure node — the independent repeats of the
            // iid. Sharing one NodeId across the materialised joint fields is safe:
            // lowering reads the constructor, it does not mutate it per field.
            Ok(vec![mm; n])
        }
        Some("joint") => {
            let c = expect_builtin_call(m, inner_resolved, "joint")
                .ok_or_else(|| refuse(inner_resolved, m, "expected joint"))?;
            if !c.named.is_empty() {
                return Err(refuse(
                    inner_resolved,
                    m,
                    "relabel over a KEYWORD joint (already field-keyed) — project it by name \
                     directly rather than relabeling — refuse rather than mislower",
                ));
            }
            if c.args.len() < 2 {
                return Err(refuse(
                    inner_resolved,
                    m,
                    "joint needs at least 2 components",
                ));
            }
            Ok(c.args.to_vec())
        }
        Some("jointchain") => Err(refuse_jointchain_projection(m, inner_resolved)),
        other => Err(refuse(
            inner_resolved,
            m,
            &format!(
                "relabel base `{}` is not an INDEPENDENT index-keyed product (iid / positional \
                 joint) — no closed-form field-keyed marginal here — refuse rather than mislower",
                other.unwrap_or("<non-builtin>")
            ),
        )),
    }
}

/// Refuse if any DROPPED field of a field-keyed product (a component in
/// `named` whose name is NOT in the selected `fields`) is not a CONFIRMED
/// `Mass::Normalized` probability measure.
///
/// §06 case 2's closed-form marginal ("the unselected components integrate to
/// 1 and drop") is an identity of an INDEPENDENT PRODUCT OF PROBABILITY
/// MEASURES: `∫ M_dropped = 1` only when `M_dropped`'s total mass is exactly 1.
/// An unnormalized dropped component — `weighted(c, M)` with `c ≠ 1`
/// (`Mass::Finite`), an unbounded `Lebesgue` (`Mass::LocallyFinite`), or an
/// unresolved/`%deferred` mass (`None`/`Mass::Unknown`/`Mass::Deferred`) —
/// integrates to its OWN total mass, not 1; dropping it unchecked would
/// silently omit that mass as a multiplicative (additive in log-space) factor
/// from the marginal. Refuse rather than mislower (a wrong marginal is the
/// worst outcome) — mirroring [`lower_normalize`]'s own reading of the mass
/// via `resolve_ref_one` + `Module::type_of`.
///
/// `measure_of` extracts the scored measure node from a field's raw value:
/// for a keyword `joint` the value IS the measure; for a record-of-draws it is
/// `draw(Mₐ)` (or a ref to one), unwrapped via [`resolve_component_draw`]. A
/// field whose value does not resolve to a recognisable measure component is
/// left to the caller's own subsequent match (e.g. [`select_projection_fields`]
/// / the density dispatcher) to refuse.
fn refuse_unnormalized_dropped_fields(
    m: &Module,
    node: NodeId,
    named: &[NamedArg],
    fields: &[Box<str>],
    measure_of: impl Fn(&Module, NodeId) -> Option<NodeId>,
) -> Result<(), RefuseError> {
    for field in named {
        if field.kind != NamedKind::Field {
            continue; // a non-field named arg is refused elsewhere by the caller
        }
        let name = m.resolve(field.name);
        if fields.iter().any(|f| f.as_ref() == name) {
            continue; // selected — kept, not dropped
        }
        let Some(measure) = measure_of(m, field.value) else {
            continue;
        };
        let (resolved, _) = resolve_ref_one(m, measure);
        let mass = match m.type_of(resolved) {
            Some(Type::Measure { mass, .. }) => Some(*mass),
            _ => None,
        };
        if mass != Some(Mass::Normalized) {
            return Err(refuse(
                node,
                m,
                &format!(
                    "projection drops a non-normalized component (field `{name}`); the marginal \
                     is not closed-form here (§06 case 2 requires each dropped component to be a \
                     normalized probability measure) — refuse rather than mislower",
                ),
            ));
        }
    }
    Ok(())
}

/// Pick, in `fields` order, the `%field` component of `source` whose name
/// resolves to each selected field string. Refuse (rather than silently drop) a
/// selected field with no matching component — a projection that names a field
/// the product does not carry is malformed.
///
/// Also refuse a DUPLICATE selected field (`get(_, ["a", "a"])`): building a
/// sub-joint/sub-record with the same `%field a` twice would make
/// [`lower_keyword_joint`] / [`lower_record_of_draws`] sum ONE density term
/// per named entry — double-counting `a`'s contribution to the marginal.
/// Refuse rather than silently double-count.
fn select_projection_fields(
    m: &Module,
    node: NodeId,
    source: &[NamedArg],
    fields: &[Box<str>],
) -> Result<Vec<NamedArg>, RefuseError> {
    let mut seen: Vec<&str> = Vec::with_capacity(fields.len());
    let mut out = Vec::with_capacity(fields.len());
    for f in fields {
        if seen.contains(&f.as_ref()) {
            return Err(refuse(
                node,
                m,
                &format!(
                    "projection selects a duplicate field `{f}` — the sub-product would score it \
                     twice, double-counting its density term — refuse rather than mislower"
                ),
            ));
        }
        seen.push(f.as_ref());
        let found = source
            .iter()
            .find(|n| n.kind == NamedKind::Field && m.resolve(n.name) == f.as_ref())
            .ok_or_else(|| {
                refuse(
                    node,
                    m,
                    &format!(
                        "projection selects field `{f}` that is not a component of the product \
                         measure — refuse rather than mislower"
                    ),
                )
            })?;
        out.push(*found);
    }
    Ok(out)
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
///
/// Shared with `sample::lower_draw`'s `iid(K, n)` sample fan-out, which needs
/// the identical static-length read before batching a `builtin_sample` over
/// `n` — see that call site for why it reuses this rather than re-deriving it.
pub(crate) fn iid_static_size(m: &Module, iid_node: NodeId) -> Option<usize> {
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
/// `None`).
/// `N == 0` is the empty independent product: Σ over an empty index set is 0, so
/// it lowers to the log-density literal `0.0` (consistent with the empty measure
/// `record()`), not a refusal.
///
/// **Axis-native fast path for a PRIMITIVE `M`.** When `M` is a bare
/// distribution constructor (`split_kernel_constructor` succeeds — e.g.
/// `Normal(mu = a, sigma = b)`), the Σ is emitted as ONE axis-level expression
/// — `sum(broadcast(builtin_logdensityof, K, broadcast(record, p = vector(a),
/// …), v))` — identical for `N = 3` or `N = 3000`, rather than `N` unrolled
/// `get0`/`add` terms (see [`emit_kernel_broadcast_density`], shared with
/// [`lower_broadcast_kernel`]'s value-broadcast case). `M`'s params are a
/// length-1 ARRAY-of-records (a bare record is not a legal broadcast input,
/// §04 "Broadcasting", "Disallowed inputs") that singleton-expands across the
/// obs axis, since a primitive `iid`'s per-copy kernel is the SAME
/// distribution at every index (unlike a value-broadcast's per-cell param
/// arrays). Each scalar param is lifted to a length-1 `vector(param)` first —
/// `vector(record(…))` directly is rejected by §03's array-literal-element
/// restriction, so the array-of-records is synthesized the same way
/// [`lower_broadcast_kernel`] does, via `broadcast(record, …)` over length-1
/// arrays rather than the `vector` literal constructor.
///
/// **Composed-`M` fallback: static unroll.** When `M` is not a bare
/// constructor (a `joint`, `pushfwd`, nested `iid`, …), `split_kernel_constructor`
/// fails and this falls back to the `get0`/`fold_add` unroll below (corpus N
/// small).
///
/// **Axis-native functionof-broadcast was tried and REJECTED for this case —
/// StableHLO cannot lower it.** The natural generalization of the primitive
/// fast path — lower `density(M, x)` once against a fresh `%local`
/// placeholder `x`, wrap it in a one-input `functionof`, and
/// `sum(broadcast(functionof(...), v))` — round-trips fine through
/// `flatppl-js` (scipy oracle == exact) but crashes `crates/stablehlo`'s
/// emitter: `Emitter::lower_broadcast_userfn`
/// (`crates/stablehlo/src/emitter.rs:2296`) binds the placeholder to the
/// *whole* collection value (rank `[N, …M-shape]`) and relies on the body's
/// own arithmetic auto-broadcasting against it — a fusion model that only
/// works for a SIMPLE scalar-arithmetic body. A COMPOSED `M`'s density body
/// contains its own nested `sum(broadcast(...))` (e.g. `M = iid(Normal, k)`'s
/// own primitive fast path), whose inner `broadcast_pair` then reconciles its
/// own rank-1 kernel-param shape against the OUTER placeholder still bound to
/// the full multi-row collection — a genuine rank mismatch, not a missing
/// `broadcast_in_dim`: `Emitter::broadcast_pair` (emitter.rs:1527) panics
/// (`rank mismatch ([Some(3), Some(2)] vs [Some(1)])`) rather than silently
/// mislowering. This is an architectural gap (no notion of "evaluate the body
/// once per top-level slice" in `lower_broadcast_userfn`), not a one-line
/// fix, so the composed case keeps the unroll below rather than shipping a
/// form that fails the StableHLO leg of the three-way gate. Reproduced with
/// `logdensityof(iid(iid(Normal(mu = 0.0, sigma = 1.0), 2), 3), <shape-[3,2]
/// literal>)`; fixing `lower_broadcast_userfn` to evaluate a reified body
/// once per top-level slice (rather than once against the whole collection)
/// is an open `crates/stablehlo` follow-up, not attempted here.
///
/// **No scalar-`M` guard — deliberate asymmetry with [`lower_joint`].** `iid(M,
/// size)` is the product `M^⊗N` over ARRAYS of shape `size`, i.e. a NESTED variate
/// with a leading repeat axis `[N, …M-shape]` (§06 "Independent composition", the
/// `iid` bullet). So `get0(v, i)` recovers the full i-th
/// `M`-variate (an entire row), which is exactly what this rule scores `M` at —
/// correct for ANY `M`, scalar or not (a non-scalar `M`, e.g. `iid(MvNormal, n)`
/// or a nested `iid(iid(…), n)`, lowers correctly via the unroll fallback, its
/// inner variate reached by a further `get0`). `joint`, by contrast, has a
/// HETEROGENEOUS variate: the flat
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

    // Primitive-kernel fast path: M is a bare distribution constructor
    // (`split_kernel_constructor` succeeds). Emit the axis-native broadcast form
    // — a length-1 array-of-params-record singleton-expanded across the obs
    // axis `v`, summed. This is the card's target and reuses the tested
    // `lower_broadcast_kernel` tail.
    if let Some((ctor_sym, kwargs)) = split_kernel_constructor(m, m_inner) {
        // `broadcast` disallows a bare record/tuple input (§04 "Broadcasting",
        // "Disallowed inputs"): the params must be a COLLECTION. Building
        // `vector(record(...))` directly does NOT work — §03 forbids a record
        // as an array LITERAL element (`vector_type` in
        // `flatppl-infer::ops::vector_type` rejects it outright), so a
        // `vector`-of-records is not constructible that way. Instead reuse
        // exactly the mechanism [`lower_broadcast_kernel`] uses to produce its
        // (real, multi-cell) array-of-records: `broadcast(record, p0 = arr0,
        // …)` synthesizes `Array{elem: Record}` structurally in
        // `broadcast_type`, which is NOT subject to the literal-vector element
        // restriction. Each scalar param is first lifted to a length-1
        // `vector(param)`, so `broadcast(record, …)` sees length-1 array
        // arguments and produces a length-1 array-of-records that
        // singleton-expands across the obs axis (§04 "Size-one array axes are
        // implicitly expanded by repetition to match the size of the other
        // collection arguments") — the same per-copy kernel at every index.
        let broadcast_sym = m.intern("broadcast");
        let record_head = {
            let record_sym = m.intern("record");
            m.alloc(Node::Const(record_sym))
        };
        let record_kwargs: Vec<NamedArg> = kwargs
            .iter()
            .map(|&(name, value)| NamedArg {
                kind: NamedKind::Kwarg,
                name,
                value: build_call(m, "vector", &[value]),
            })
            .collect();
        let kernel_input = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(broadcast_sym),
            args: vec![record_head].into(),
            named: record_kwargs.into(),
            inputs: None,
        }));
        return Ok(emit_kernel_broadcast_density(m, ctor_sym, kernel_input, v));
    }

    // Composed inner measure (joint / pushfwd / nested iid): temporary fallback to
    // the per-element `get0` unroll. The axis-native functionof-broadcast lambda
    // path was implemented and three-way tested for this case and REJECTED — see
    // the doc comment above ("Axis-native functionof-broadcast was tried and
    // REJECTED for this case") — StableHLO's `lower_broadcast_userfn` cannot
    // lower a reified body that itself contains a nested `sum(broadcast(...))`
    // (a genuine rank-mismatch panic in `Emitter::broadcast_pair`, not a missing
    // `broadcast_in_dim`). Keep unrolling here until that emitter gap is fixed.
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

    // Ordered constructor parameter names (spec §08 distributions + §06
    // fundamental measures). `None` ⇒ the head is not a known measure
    // constructor (e.g. a deterministic op) ⇒ refuse.
    let param_names =
        flatppl_infer::constructor_param_names(m.resolve(ctor_sym)).ok_or_else(|| {
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
    Ok(emit_kernel_broadcast_density(
        m,
        ctor_sym,
        kernel_inputs,
        obs,
    ))
}

/// Emit `sum(broadcast(builtin_logdensityof, K, kernel_input, obs))` — the
/// axis-native kernel-broadcast density tail. `kernel_input` is the array of
/// per-cell constructor records — a length-`n` array for a value-broadcast, or a
/// length-1 array for `iid`'s primitive fast path, whose size-1 axis broadcast
/// replicates across the obs axis (a bare record is NOT a legal broadcast input,
/// §04; see [`lower_iid`]). Shared by [`lower_broadcast_kernel`] and
/// [`lower_iid`]'s primitive fast path.
fn emit_kernel_broadcast_density(
    m: &mut Module,
    ctor_sym: Symbol,
    kernel_input: NodeId,
    obs: NodeId,
) -> NodeId {
    let broadcast_sym = m.intern("broadcast");
    let kernel = m.alloc(Node::Const(ctor_sym));
    let ldo_head = {
        let ldo_sym = m.intern("builtin_logdensityof");
        m.alloc(Node::Const(ldo_sym))
    };
    let per_cell = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(broadcast_sym),
        args: vec![ldo_head, kernel, kernel_input, obs].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }));
    build_call(m, "sum", &[per_cell])
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
/// silently dropped.
///
/// Keyword `joint(name₁ = M₁, …)` (named components → RECORD variate) shares
/// the `joint` op name with the positional form, so it also reaches this
/// function — its components live in `named` rather than `args`, so the
/// dispatch below reads which of `args`/`named` is populated and routes to
/// [`lower_keyword_joint`] for the record form. A call with BOTH populated
/// (mixing positional and keyword components) is neither form — refused,
/// rather than guessing which one was meant.
fn lower_joint(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (positional, named): (Vec<NodeId>, Vec<NamedArg>) = {
        let c = expect_builtin_call(m, node, "joint")
            .ok_or_else(|| refuse(node, m, "expected joint"))?;
        (c.args.to_vec(), c.named.to_vec())
    };

    if !positional.is_empty() && !named.is_empty() {
        return Err(refuse(
            node,
            m,
            "joint mixes positional and keyword components; a joint is either the \
             positional cat-variate form or the keyword record-variate form, not both",
        ));
    }

    if !named.is_empty() {
        return lower_keyword_joint(m, node, &named, v);
    }

    let inner = positional;
    if inner.len() < 2 {
        return Err(refuse(node, m, "joint needs at least 2 components"));
    }
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

/// `logdensityof(joint(name₁ = M₁, …, nameₖ = Mₖ), v)` = `Σᵢ
/// logdensityof(Mᵢ, v.nameᵢ)` — the keyword/record form of `joint` (§04
/// example: `prior = joint(theta1 = Normal(...), theta2 = Exponential(...))`;
/// §06 "joint and iid (independent products)"). The variate `v` is a RECORD
/// keyed by the SAME field names as the joint's named components — unlike the
/// positional form's flat `cat` vector, so there is no `get0`-slicing and
/// (consequently) no scalar-component restriction: each component is matched
/// to its OWN record field by name and scored there directly, whatever shape
/// that field's value has. The downstream `lower_measure_density` /
/// `build_density_term` recursion already domain-checks each component
/// against its own pinned field value, which is exactly the guard a
/// non-scalar component needs — no upfront kind inspection required (contrast
/// [`lower_joint`]'s positional path, which DOES need one because `get0(v,
/// i)`'s value infers to `%deferred` and so bypasses that same check).
///
/// Called only from [`lower_joint`] once it has confirmed `named` is
/// non-empty and `args` is empty, so `named` here is always non-empty.
///
/// **Refuses** (rather than mislowering) when: a named component is not a
/// `%field` (a malformed named arg); the value `v` is not a `record(...)`
/// node; `v`'s record carries a positional (non-named) element alongside its
/// named fields; or `v`'s record is missing a field that one of the joint's
/// named components expects.
///
/// An extra value-record field NOT named by the joint (e.g. `record(x=.., y=..,
/// z=..)` against `joint(x=.., y=..)`) is silently ignored — this matches the
/// pre-existing leniency of the record-of-draws helper
/// ([`match_independent_record`]) and is not tightened here.
fn lower_keyword_joint(
    m: &mut Module,
    node: NodeId,
    named: &[NamedArg],
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    for n in named {
        if n.kind != NamedKind::Field {
            return Err(refuse(
                node,
                m,
                "non-field named arg in keyword joint (expected `name = measure` components)",
            ));
        }
    }

    // As in `match_independent_record`: the scored value may be a ref to a record
    // binding (`theta = record(...)`), not an inline `record(...)`. Resolve one
    // ref level before destructuring.
    let (v, _) = resolve_ref_one(m, v);
    let vrec = expect_builtin_call(m, v, "record")
        .ok_or_else(|| refuse(v, m, "joint value must be a record"))?;
    // A stray positional element mixed with the named fields (e.g. `record(0.9,
    // x = 0.5, y = 1.0)`) is not a well-formed field-keyed value record — refuse
    // rather than silently drop the positional slot, mirroring the equivalent
    // guard on `match_independent_record` ("value record with positional args").
    if !vrec.args.is_empty() {
        return Err(refuse(v, m, "joint value record carries positional args"));
    }
    let vrec_named: Vec<NamedArg> = vrec.named.to_vec();

    let mut terms = Vec::with_capacity(named.len());
    for field in named {
        let pinned = lookup_field(m, &vrec_named, field.name).ok_or_else(|| {
            refuse(
                v,
                m,
                &format!(
                    "missing field {} in joint value record",
                    m.resolve(field.name)
                ),
            )
        })?;
        terms.push(lower_measure_density(m, field.value, pinned)?);
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

/// Read a primitive constructor call into its constructor symbol and the
/// arguments as `(param_name, value)` pairs, accepting BOTH surface calling
/// conventions (spec §04 "Calling conventions": every built-in ordinary
/// callable has a defined input order and accepts positional and keyword
/// arguments):
///
/// * keyword form `Ctor(mu = m, sigma = s)` → the named args verbatim;
/// * positional form `Ctor(m, s)` → each positional arg bound to the
///   constructor's ordered parameter name (§08 distribution / §06
///   fundamental-measure parameter order, via
///   [`flatppl_infer::constructor_param_names`]);
/// * mixed form `Ctor(m, sigma = s)` → positional args bound by order, then the
///   keyword args.
///
/// This only *reads* the call; it does not rewrite `node` — the surface form
/// (positional preferred for term-rewriting) is left intact. The name mapping
/// exists so the caller can build the by-name `record` kernel_input that
/// `builtin_logdensityof` / `builtin_sample` require (a `record` takes named
/// fields, §04).
///
/// `None` (⇒ the caller refuses, per refuse-don't-mislower) if `node` is not a
/// `Call`, not builtin-headed, has a non-kwarg named arg, or — when positional
/// args are present — the head is not a known distribution constructor, carries
/// more positional args than the constructor has parameters, or binds a
/// parameter both positionally and by keyword (a §04 double-bind).
///
/// Shared by [`build_density_term`] (the density-side kernel/kernel_input build)
/// and the sample-side leaf (`sample::split_constructor`) — both need exactly
/// this constructor-symbol-plus-arguments read before building their respective
/// `builtin_*` call. Needs `&mut` only to intern parameter-name symbols for the
/// positional mapping; the keyword-only fast path is non-mutating.
pub(crate) fn split_kernel_constructor(
    m: &mut Module,
    node: NodeId,
) -> Option<(Symbol, Vec<(Symbol, NodeId)>)> {
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    // Snapshot positional args and keyword args (NodeId/Symbol are Copy) so the
    // `&Call` borrow ends before interning parameter names below needs `&mut`.
    let pos_args: Vec<NodeId> = c.args.to_vec();
    let mut kwargs = Vec::with_capacity(c.named.len());
    for n in c.named.iter() {
        if n.kind != NamedKind::Kwarg {
            return None;
        }
        kwargs.push((n.name, n.value));
    }

    // Keyword-only form (the common case): return the named args verbatim.
    if pos_args.is_empty() {
        return Some((sym, kwargs));
    }

    // Positional / mixed form: bind positional args to the constructor's ordered
    // parameter names. A head that is not a known distribution constructor, or
    // more positional args than it has parameters, is not a well-formed
    // positional constructor call → None.
    let param_names = flatppl_infer::constructor_param_names(m.resolve(sym))?;

    // §04 "Calling conventions" auto-splatting: `Ctor(record(p1 = …, p2 = …))`
    // is equivalent to `Ctor(p1 = …, p2 = …)`. When the sole positional arg is
    // a RECORD whose field names match the constructor's parameter names,
    // distribute its fields across the params instead of (mis)binding the whole
    // record to `param_names[0]` and dropping the rest. §04 keys auto-splat off
    // the argument's TYPE, not its surface syntax, so this fires for both a
    // literal `record(…)` and an opaque multi-output call returning a record
    // (e.g. `gamma_shape_rate(μ, σ)` whose body is `record(shape = …, rate =
    // …)`) — the latter can't be read field-wise from the node, so each field
    // is pulled with `get(arg, "field")` (the same lowering as `arg.field`,
    // §07). Regression for buffy #247. A single positional arg that is NOT a
    // param-matching record falls through to positional index-binding below.
    // §04 auto-splat applies at ANY arity: a positional `record(...)` whose
    // field names match the callable's parameter names binds those fields to
    // the parameters, so `Dirac(record(value = v))` is `Dirac(value = v)` (a
    // point mass at `v`, NOT at the record). The record-VALUE form is the
    // keyword `Dirac(value = record(...))`, which is not a positional splat and
    // reaches the index-binding path below. Inference resolves the same way (it
    // splats a positional record for §08 distributions and, via the
    // fundamental-measure arms, for `Dirac`/`Lebesgue`/`Counting`), so the two
    // engines stay in step.
    if kwargs.is_empty() && pos_args.len() == 1 {
        let arg = pos_args[0];
        let record_field_names: Option<Vec<String>> = match m.type_of(arg) {
            Some(Type::Record(fields)) => Some(
                fields
                    .iter()
                    .map(|(s, _)| m.resolve(*s).to_string())
                    .collect(),
            ),
            _ => None,
        };
        if let Some(field_names) = record_field_names {
            let params: std::collections::BTreeSet<&str> =
                param_names.iter().map(String::as_str).collect();
            let fields: std::collections::BTreeSet<&str> =
                field_names.iter().map(String::as_str).collect();
            if params == fields {
                let mut args: Vec<(Symbol, NodeId)> = Vec::with_capacity(param_names.len());
                for p in &param_names {
                    let key = m.alloc(Node::Lit(Scalar::Str(p.clone().into_boxed_str())));
                    let getter = build_call(m, "get", &[arg, key]);
                    args.push((m.intern(p), getter));
                }
                return Some((sym, args));
            }
        }
    }

    if pos_args.len() > param_names.len() {
        return None;
    }
    let mut args: Vec<(Symbol, NodeId)> = Vec::with_capacity(pos_args.len() + kwargs.len());
    for (i, &arg) in pos_args.iter().enumerate() {
        args.push((m.intern(&param_names[i]), arg));
    }
    // A parameter bound BOTH positionally and by keyword is a §04 double-bind
    // (static error) — refuse rather than emit a record with duplicate fields.
    if kwargs
        .iter()
        .any(|(kw_name, _)| args.iter().any(|(pos_name, _)| pos_name == kw_name))
    {
        return None;
    }
    args.extend(kwargs);
    Some((sym, args))
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
            "primitive measure must be a built-in constructor call with well-formed positional or keyword arguments",
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
// Helper: reduce a logdensityof measure argument to its underlying measure
// ---------------------------------------------------------------------------

/// Reduce a `logdensityof` measure argument to the measure expression the
/// recursive dispatcher expects.
///
/// The measure argument comes in two shapes that both reduce to "the underlying
/// measure":
///
/// * `lawof(M_value)` — `lawof` reifies a (stochastic) value to its law; we
///   score the value's law, i.e. `M_value` (a record-of-draws, a combinator,
///   …). This is the inline form the Task-3/4 record/combinator goldens use.
/// * a bare measure expression — e.g. `(%ref self pp)` where `pp = kchain(…)`
///   (or any combinator binding), or a grafted cross-module measure handle
///   (`logdensityof(m.d, v)`). Here the measure is already a measure; there is
///   no `lawof` wrapper to strip.
///
/// We resolve one level of ref indirection and strip a `lawof` if present;
/// otherwise we hand the (original, unresolved) measure node straight to the
/// dispatcher, which itself resolves one ref level and dispatches by op.
///
/// Takes the measure argument node directly (not the enclosing `logdensityof`
/// query), so it works on a grafted node the caller substituted for the original
/// cross-module ref.
fn measure_of_arg(m: &Module, measure_arg: NodeId) -> Result<NodeId, RefuseError> {
    let (resolved, _) = resolve_ref_one(m, measure_arg);
    if let Some(law) = expect_builtin_call(m, resolved, "lawof") {
        if law.args.len() != 1 {
            return Err(refuse(resolved, m, "lawof expects 1 arg"));
        }
        return Ok(law.args[0]);
    }
    Ok(measure_arg)
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
pub(crate) fn fold_add(m: &mut Module, terms: &[NodeId]) -> NodeId {
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
