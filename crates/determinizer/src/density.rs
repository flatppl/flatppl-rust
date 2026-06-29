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
//! - `weighted(w, M)` → `add(log(w), density(M, v))`
//! - `logweighted(ℓ, M)` → `add(ℓ, density(M, v))`
//! - `superpose(M₁, …, Mₖ)` → `logsumexp(density(M₁, v), …, density(Mₖ, v))`
//! - `normalize(M)` → `sub(density(M, v), log(totalmass(M)))`
//! - `truncate(M, S)` → `ifelse(elementof(v, S), density(M, v), neg(inf))`
//! - `pushfwd(bijection(f, f_inv, logvol), M)` → `sub(density(M, f_inv(v)), logvol(f_inv(v)))`
//!
//! **Refused:** `kchain` marginals, `joint`/`iid`, `bayesupdate`, `disintegrate`, `restrict`,
//! `likelihoodof`, `pushfwd` with a non-bijection argument, and any unrecognised shape.

use crate::refuse::RefuseError;
use flatppl_core::{
    BindingId, Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Symbol,
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
    let (measure_expr, v) = extract_logdensityof_args(m, query)?;
    lower_measure_density(m, measure_expr, v)
}

// ---------------------------------------------------------------------------
// Core recursive dispatcher
// ---------------------------------------------------------------------------

/// Compute the log-density of `measure_expr` at `v`, returning a deterministic node.
/// `measure_expr` may be a record-of-draws, a combinator, a `(%ref self x)` pointing
/// to one of those, or a bare primitive constructor.
fn lower_measure_density(
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
        // Refused combinators — refused here rather than mis-lowered.
        Some("joint") | Some("iid") | Some("kchain") | Some("markovchain") | Some("kscan")
        | Some("jointchain") | Some("bayesupdate") | Some("disintegrate") | Some("restrict")
        | Some("likelihoodof") | Some("locscale") => Err(refuse_op(measure_node, m)),
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

    if components.is_empty() {
        return Err(refuse(record_node, m, "empty measure record"));
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

/// `logdensityof(weighted(w, M), v)` = `log(w) + logdensityof(M, v)`
fn lower_weighted(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let c = expect_builtin_call(m, node, "weighted")
        .ok_or_else(|| refuse(node, m, "expected weighted"))?;
    if c.args.len() != 2 {
        return Err(refuse(node, m, "weighted expects 2 args"));
    }
    let w_node = c.args[0];
    let m_inner = c.args[1];

    let inner_density = lower_measure_density(m, m_inner, v)?;
    let log_w = build_call(m, "log", &[w_node]);
    Ok(build_call(m, "add", &[log_w, inner_density]))
}

/// `logdensityof(logweighted(ℓ, M), v)` = `ℓ + logdensityof(M, v)`
fn lower_logweighted(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let c = expect_builtin_call(m, node, "logweighted")
        .ok_or_else(|| refuse(node, m, "expected logweighted"))?;
    if c.args.len() != 2 {
        return Err(refuse(node, m, "logweighted expects 2 args"));
    }
    let lw_node = c.args[0];
    let m_inner = c.args[1];

    let inner_density = lower_measure_density(m, m_inner, v)?;
    Ok(build_call(m, "add", &[lw_node, inner_density]))
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

    Ok(build_variadic_call(m, "logsumexp", &density_terms))
}

/// `logdensityof(normalize(M), v)` = `logdensityof(M, v) - log(totalmass(M))`
fn lower_normalize(m: &mut Module, node: NodeId, v: NodeId) -> Result<NodeId, RefuseError> {
    let m_inner = {
        let c = expect_builtin_call(m, node, "normalize")
            .ok_or_else(|| refuse(node, m, "expected normalize"))?;
        if c.args.len() != 1 {
            return Err(refuse(node, m, "normalize expects 1 arg"));
        }
        c.args[0]
    };

    let inner_density = lower_measure_density(m, m_inner, v)?;
    let totalmass_node = build_call(m, "totalmass", &[m_inner]);
    let log_totalmass = build_call(m, "log", &[totalmass_node]);
    Ok(build_call(m, "sub", &[inner_density, log_totalmass]))
}

/// `logdensityof(truncate(M, S), v)` = `ifelse(elementof(v, S), logdensityof(M, v), neg(inf))`
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
    let elementof = build_call(m, "elementof", &[v, s_node]);
    let inf_sym = m.intern("inf");
    let inf_node = m.alloc(Node::Const(inf_sym));
    let neg_inf = build_call(m, "neg", &[inf_node]);
    Ok(build_call(
        m,
        "ifelse",
        &[elementof, inner_density, neg_inf],
    ))
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

// ---------------------------------------------------------------------------
// Primitive distribution constructor density term (Task 3 helper)
// ---------------------------------------------------------------------------

/// Build `builtin_logdensityof(kernel, kernel_input, pinned)` for a primitive
/// distribution constructor `measure` applied to keyword arguments.
fn build_density_term(
    m: &mut Module,
    measure: NodeId,
    pinned: NodeId,
) -> Result<NodeId, RefuseError> {
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

fn extract_logdensityof_args(m: &Module, query: NodeId) -> Result<(NodeId, NodeId), RefuseError> {
    let q = expect_builtin_call(m, query, "logdensityof")
        .ok_or_else(|| refuse(query, m, "expected logdensityof"))?;
    if q.args.len() != 2 {
        return Err(refuse(query, m, "logdensityof expects 2 args"));
    }
    let law_arg = q.args[0];
    let v = q.args[1];

    let law = expect_builtin_call(m, law_arg, "lawof")
        .ok_or_else(|| refuse(law_arg, m, "logdensityof first arg must be lawof(...)"))?;
    if law.args.len() != 1 {
        return Err(refuse(law_arg, m, "lawof expects 1 arg"));
    }
    Ok((law.args[0], v))
}

// ---------------------------------------------------------------------------
// Utility: IR construction helpers
// ---------------------------------------------------------------------------

/// Allocate a positional builtin call `head(args…)`.
fn build_call(m: &mut Module, head: &str, args: &[NodeId]) -> NodeId {
    let sym = m.intern(head);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: args.to_vec().into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }))
}

/// Allocate a variadic positional builtin call `head(args…)` (same as `build_call`; alias for clarity).
fn build_variadic_call(m: &mut Module, head: &str, args: &[NodeId]) -> NodeId {
    build_call(m, head, args)
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
fn expect_builtin_call<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a Call> {
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
fn resolve_ref_one(m: &Module, id: NodeId) -> (NodeId, Option<BindingId>) {
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
fn refuse(id: NodeId, m: &Module, reason: &str) -> RefuseError {
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
