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
//!   constant/scalar (used as-is) OR a **function of the variate** (§06:469), in
//!   which case it is **applied at the variate**: `log w(v)`. When `w`'s body
//!   contains measure ops (e.g. an inner `logdensityof`), the driver's subtree
//!   scan finds and lowers them on a later iteration and the plain-function call
//!   is beta-reduced by the backend.
//! - `logweighted(ℓ, M)` → `add(ℓ(v), density(M, v))` — likewise, `ℓ` may be a
//!   constant/scalar or a function of the variate, applied as `ℓ(v)` (already in
//!   log space, so no outer `log`).
//! - `superpose(M₁, …, Mₖ)` → `logsumexp(density(M₁, v), …, density(Mₖ, v))`
//! - `normalize(M)` → `density(M, v)` when `M` is already a probability measure
//!   (closed-form `logZ = 0`); when `M = truncate(base, interval(lo, hi))` with
//!   `base` a primitive univariate constructor, → `sub(density(M, v),
//!   log(sub(touniform(base, hi), touniform(base, lo))))` — the closed-form
//!   `Z = CDF(hi) - CDF(lo)` via the `builtin_touniform` CDF transport
//!   (§6a:179, §06:471); **refuses** for any other unnormalized `M` (no
//!   closed-form mass rule; `totalmass` is OUT of FlatPDL).
//! - `truncate(M, S)` → `ifelse(in(v, S), density(M, v), neg(inf))` (the `_ in R`
//!   membership builtin, which infers to a boolean — `elementof` is a set-valued
//!   parameter declaration, not a membership predicate).
//! - `pushfwd(bijection(f, f_inv, logvol), M)` → `sub(density(M, f_inv(v)), logvol(f_inv(v)))`
//! - `iid(M, N)` → `Σ_{i<N} density(M, get0(v, i))` — **`N` must be a literal
//!   integer** (static unroll; corpus `N` is small). A non-literal `N` is
//!   **refused**.
//! - `joint(M₁,…,Mₖ)` (**positional only**) → `Σᵢ density(Mᵢ, get0(v, i))` —
//!   **scalar-variate components only**; a non-scalar component variate
//!   refuses via the recursive call's own domain/variate shape check. Keyword
//!   `joint(name = M, …)` (named components → record variate) shares the
//!   `joint` op name so it reaches the same dispatch arm, but its components
//!   live in `named` with an empty positional `args`, so it is **refused**
//!   rather than mislowered.
//!
//! **Likelihood query** (audit H2): `logdensityof(likelihoodof(K, obs), θ)` is
//! handled at the `logdensityof` *entry* (not via `lower_measure_density`). Its
//! arg2 is the parameter point θ (a record), NOT the variate; the variate is the
//! `obs` baked into the likelihood. Each θ field is bound to the matching module
//! parameter binding, then `K` is scored at `obs` — §06:492
//! `densityof(likelihoodof(K, obs), θ) = pdf(κ(θ), obs)`.
//!
//! **Refused:** `kchain` marginals, keyword `joint`, `bayesupdate`, `disintegrate`,
//! `restrict`, `pushfwd` with a non-bijection argument, `iid` with a non-literal
//! size, `joint` with a non-scalar component variate, and any unrecognised shape.
//! (`likelihoodof` reaching `lower_measure_density` still refuses there as a
//! safety net — it is normally unwrapped at the `logdensityof` entry above.)

use crate::refuse::RefuseError;
use flatppl_core::{
    BindingId, Call, CallHead, Mass, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Scalar,
    Symbol, Type,
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
    let (resolved, _) = resolve_ref_one(m, arg1);
    if is_likelihood(m, arg1) || builtin_name(m, resolved) == Some("likelihoodof") {
        return lower_likelihood_query(m, resolved, arg2);
    }
    // Measure query: arg2 is the variate (existing path).
    let (measure_expr, v) = extract_logdensityof_args(m, query)?;
    lower_measure_density(m, measure_expr, v)
}

/// True iff `id` infers to a `Likelihood` type.
fn is_likelihood(m: &Module, id: NodeId) -> bool {
    matches!(m.type_of(id), Some(Type::Likelihood { .. }))
}

/// `logdensityof(likelihoodof(K, obs), θ)` = density of `K` at the observed `obs`,
/// with `K`'s free parameters bound from the θ record (§06:492, audit H2).
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
    bind_params_from_record(m, theta)?;
    lower_measure_density(m, k, obs)
}

/// Bind each `%field name = value` in the θ record to the module binding `name`
/// (parameterized→fixed), so `K`'s parameter refs resolve to the scored point.
fn bind_params_from_record(m: &mut Module, theta: NodeId) -> Result<(), RefuseError> {
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
    for (name, value) in fields {
        let bid = m
            .binding_by_name(name)
            .ok_or_else(|| refuse(value, m, "θ names a parameter with no module binding"))?;
        m.set_binding_rhs(bid, value);
    }
    Ok(())
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
        Some("joint") => lower_joint(m, measure_node, v),
        // Refused combinators — refused here rather than mis-lowered.
        Some("markovchain") | Some("kscan") | Some("jointchain") | Some("bayesupdate")
        | Some("disintegrate") | Some("restrict") | Some("likelihoodof") | Some("locscale") => {
            Err(refuse_op(measure_node, m))
        }
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

    // Empty record (degenerate joint): log-density is 0 (§10 item 5).
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
/// is a constant/scalar OR a function of the variate (§06:469). A constant/scalar
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
/// is a constant/scalar OR a function of the variate (§06:469). The log-weight is
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

/// `logdensityof(superpose(M₁, …, Mₖ), v)` = `logsumexp(density(M₁,v), …, density(Mₖ,v))`
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

    Ok(build_call(m, "logsumexp", &density_terms))
}

/// `logdensityof(normalize(M), v)` = `logdensityof(M, v) - logZ`, where
/// `Z = totalmass(M)` must be a **closed-form** deterministic expression — never
/// the `totalmass` query op, which is OUT of FlatPDL (measures are not values).
///
/// Two closed-form mass rules are available:
/// * If `M` is already a probability measure (`Type::Measure { mass:
///   Mass::Normalized, .. }`) then `Z = 1`, `logZ = 0`, so `normalize(M)` is
///   the identity on the density — it lowers to just `logdensityof(M, v)`.
/// * If `M` is `truncate(base, interval(lo, hi))` with `base` a primitive
///   univariate constructor, `Z = CDF(hi) - CDF(lo)` via the `builtin_touniform`
///   CDF transport (§6a:179, §06:471): `logZ = log(touniform(base, hi) -
///   touniform(base, lo))`. The `-inf` outside-support gate is already handled
///   by the existing `truncate` lowering, so the emitted density term is just
///   `lower_measure_density(m, m_inner, v)` minus this `logZ`.
///
/// Any other unnormalized `M` (`Finite`, `LocallyFinite`, a non-truncate
/// combinator, …) has no closed-form mass rule here, so we **refuse** rather
/// than emit `totalmass` (refuse-don't-mislower).
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
    // via the builtin_touniform (CDF) transport (§6a:179). Extract the
    // truncate/interval shape and its endpoints BEFORE any mutable builds
    // below (immutable reads of `m` must precede `m.alloc`/`build_*` calls).
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
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    if let Some((base, lo, hi)) = truncate_shape {
        let density = lower_measure_density(m, m_inner, v)?; // truncate handles the -inf gate
        let (kernel, input) = kernel_and_input(m, base)?; // helper below
        let cdf_hi = build_touniform(m, kernel, input, hi);
        let cdf_lo = build_touniform(m, kernel, input, lo);
        let z = build_call(m, "sub", &[cdf_hi, cdf_lo]);
        let log_z = build_call(m, "log", &[z]);
        return Ok(build_call(m, "sub", &[density, log_z]));
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

/// Extract `(kernel_const, kernel_input_record)` from a primitive constructor
/// `Normal(mu = …, sigma = …)` — the `builtin_*` primitive's arg 0/1 shape.
/// Resolves one level of `(%ref self x)` indirection first, so a truncation
/// base bound by name (`g = Normal(...); truncate(g, ...)`) is classified by
/// its constructor.
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

/// Read a literal non-negative integer from `id` (`Scalar::Int`, or an integral
/// `Scalar::Real`). Returns `None` if `id` is not such a literal.
fn literal_usize(m: &Module, id: NodeId) -> Option<usize> {
    match m.node(id) {
        Node::Lit(Scalar::Int(n)) if *n >= 0 => Some(*n as usize),
        Node::Lit(Scalar::Real(r)) if *r >= 0.0 && r.fract() == 0.0 => Some(*r as usize),
        _ => None,
    }
}

/// `logdensityof(iid(M, N), v)` = `Σ_{i<N} logdensityof(M, get0(v, i))` (§06:473,
/// "iid(M, n) → Σ_i log densityof(M, xᵢ)"). N comes from the `iid` count arg,
/// which must be a literal integer. Static unroll (corpus N small; broadcast+
/// reduce is the noted scale path, not built).
fn lower_iid(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let (m_inner, n) = {
        let c =
            expect_builtin_call(m, node, "iid").ok_or_else(|| refuse(node, m, "expected iid"))?;
        if c.args.len() != 2 {
            return Err(refuse(node, m, "iid expects 2 args (measure, size)"));
        }
        let n = literal_usize(m, c.args[1])
            .ok_or_else(|| refuse(c.args[1], m, "iid size must be a literal integer"))?;
        (c.args[0], n)
    };
    if n == 0 {
        return Err(refuse(node, m, "iid with zero size has no density"));
    }
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        let idx = m.alloc(Node::Lit(Scalar::Int(i as i64)));
        let elem = build_call(m, "get0", &[v, idx]);
        terms.push(lower_measure_density(m, m_inner, elem)?);
    }
    Ok(fold_add(m, &terms))
}

/// `logdensityof(joint(M₁,…,Mₖ), v)` = `Σ logdensityof(Mᵢ, get0(v, i))` (§06:473).
/// The variate is the positional `cat` of the component variates.
///
/// **Scope:** positional `joint` only, scalar-variate components. `joint`'s
/// variate is the positional `cat` of the component variates (§06:473); for
/// scalar-variate components the destructuring is `get0(v, i)`. Component
/// variates of higher rank need `cat`-slice routing, which this does not build
/// — a component whose variate is non-scalar refuses via the recursive density
/// call's own shape handling (`build_density_term`'s domain/variate kind
/// check). Keyword `joint(name₁ = M₁, …)` (named components → record variate)
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
    let mut terms = Vec::with_capacity(inner.len());
    for (i, &mi) in inner.iter().enumerate() {
        let idx = m.alloc(Node::Lit(Scalar::Int(i as i64)));
        let elem = build_call(m, "get0", &[v, idx]);
        terms.push(lower_measure_density(m, mi, elem)?);
    }
    Ok(fold_add(m, &terms))
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

pub(crate) fn build_density_term(
    m: &mut Module,
    measure: NodeId,
    pinned: NodeId,
) -> Result<NodeId, RefuseError> {
    // Refuse scoring a measure at a variate whose structural KIND clearly
    // mismatches the measure's variate domain — a scalar `Normal` scored at a
    // record / tuple / vector (review finding F4). Inference does not reject
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

    let (ctor_sym, kwargs): (Symbol, Vec<(Symbol, NodeId)>) = {
        let Node::Call(c) = m.node(measure) else {
            return Err(refuse(measure, m, "primitive measure must be a Call node"));
        };
        let CallHead::Builtin(sym) = c.head else {
            return Err(refuse(
                measure,
                m,
                "user / module-qualified constructor not yet supported",
            ));
        };
        if !c.args.is_empty() {
            return Err(refuse(
                measure,
                m,
                "primitive constructor with positional args not supported",
            ));
        }
        let mut kwargs = Vec::with_capacity(c.named.len());
        for n in c.named.iter() {
            if n.kind != NamedKind::Kwarg {
                return Err(refuse(measure, m, "non-kwarg named arg in constructor"));
            }
            kwargs.push((n.name, n.value));
        }
        (sym, kwargs)
    };

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
fn build_record(m: &mut Module, fields: &[(Symbol, NodeId)]) -> NodeId {
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
fn draw_argument(m: &Module, id: NodeId) -> Option<NodeId> {
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
fn builtin_name(m: &Module, id: NodeId) -> Option<&str> {
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
fn refuse_op(id: NodeId, m: &Module) -> RefuseError {
    let op = builtin_name(m, id).unwrap_or("unknown").to_string();
    RefuseError {
        node: id,
        construct: op.clone(),
        reason: format!(
            "density lowering for `{op}` is not implemented (deferred to a later task)"
        ),
    }
}
