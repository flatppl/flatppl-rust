//! Density disintegration — the **independent / local** case (spec §06,
//! "Density of composed measures", joint/iid → Σ of component log-densities).
//!
//! Lowers `logdensityof(lawof(x), v)` where `x` is a `record` of mutually
//! independent component measures, each ultimately a single `draw(mᵢ)`. The
//! per-component log-densities are summed:
//!
//! ```text
//! logdensityof(lawof(record(a = a, b = b)), record(a = vₐ, b = v_b))
//!   ⤳  add( builtin_logdensityof(kₐ, inputₐ, vₐ),
//!           builtin_logdensityof(k_b, input_b, v_b) )
//! ```
//!
//! where each component measure `draw(Normal(mu = .., sigma = ..))` contributes
//! `builtin_logdensityof(Normal, record(mu = .., sigma = ..), pinned)`. The
//! pinned variate flows from the matching field of `v`, NOT from re-running the
//! draw (audit H1/H3: pin, never re-materialise). The draw bindings the record
//! references are themselves pinned to their scored value by the driver, so no
//! stochastic `draw` survives.
//!
//! **Scope (this task):** the *local* case only — a `record` of direct,
//! independent `draw`s (component value is a `(%ref self x)` to a binding whose
//! RHS is `draw(mᵢ)`, or an inline `draw(mᵢ)`). Anything else — `joint` / `iid`
//! / `kchain` marginals, derived bindings between a draw and the law, the
//! weighted / superpose / normalize / truncate / pushfwd special cases — is
//! deferred to later tasks and **refused** here rather than mis-lowered.

use crate::refuse::RefuseError;
use flatppl_core::{
    BindingId, Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Symbol,
};

/// One scored component of an independent product: the component measure node
/// (a `draw`'s argument, e.g. the `Normal(..)` constructor), the pinned variate
/// value node from `v`, and — when the component reached us through a binding
/// reference — that binding, so the driver can pin it to the scored value.
struct Component {
    /// The distribution-constructor node `mᵢ` (e.g. `Normal(mu = .., sigma = ..)`).
    measure: NodeId,
    /// The matching part of `v` to score `mᵢ` at.
    pinned: NodeId,
    /// `Some(bid)` when the component is `(%ref self x)`; the draw binding `x`
    /// to pin to `pinned` (so the standalone `draw` disappears).
    draw_binding: Option<BindingId>,
}

/// Lower `logdensityof(lawof(x), v)` at `query` into a deterministic sum of
/// `builtin_logdensityof` terms, returning the new root node id (the `add`-sum,
/// or the single term when there is exactly one component). Refuses anything
/// outside the independent-`record`-of-`draw`s case.
///
/// Side effect: each component reached through a binding reference has that
/// binding's RHS redirected to the pinned value, so the now-scored latent is a
/// deterministic constant rather than a surviving `draw`.
pub(crate) fn lower_logdensityof(m: &mut Module, query: NodeId) -> Result<NodeId, RefuseError> {
    let components = match_independent_record(m, query)?;

    // Build a `builtin_logdensityof(kernel, kernel_input, pinned)` term per
    // component, then fold them with `add`.
    let mut terms: Vec<NodeId> = Vec::with_capacity(components.len());
    for comp in &components {
        terms.push(build_density_term(m, comp.measure, comp.pinned)?);
    }

    // Pin each referenced draw binding to its scored value (transitive pin, not
    // re-materialisation). Done after term construction so the measure nodes we
    // read above are still the draws.
    for comp in &components {
        if let Some(bid) = comp.draw_binding {
            m.set_binding_rhs(bid, comp.pinned);
        }
    }

    Ok(fold_add(m, &terms))
}

/// Match `logdensityof(lawof(record(...)), v)` and extract one [`Component`] per
/// record field, pairing each component measure with the matching field of `v`.
/// Refuses any shape outside the independent-`record`-of-`draw`s case.
fn match_independent_record(m: &Module, query: NodeId) -> Result<Vec<Component>, RefuseError> {
    // query := logdensityof(law_arg, v)
    let q = expect_builtin_call(m, query, "logdensityof").ok_or_else(|| refuse(query, m))?;
    if q.args.len() != 2 {
        return Err(refuse(query, m));
    }
    let law_arg = q.args[0];
    let v = q.args[1];

    // law_arg := lawof(measure_expr)
    let law = expect_builtin_call(m, law_arg, "lawof").ok_or_else(|| refuse(law_arg, m))?;
    if law.args.len() != 1 {
        return Err(refuse(law_arg, m));
    }
    let measure_expr = law.args[0];

    // measure_expr := record(%field nameᵢ valueᵢ ...) — the independent product.
    // `joint` / `iid` and indirection through a binding are deferred (Task 4+).
    let rec =
        expect_builtin_call(m, measure_expr, "record").ok_or_else(|| refuse(measure_expr, m))?;
    if !rec.args.is_empty() {
        // A `record` with positional args is not the field-keyed product we model.
        return Err(refuse(measure_expr, m));
    }

    // `v` must be a matching `record(%field nameⱼ valⱼ ...)` so we can pin by name.
    let vrec = expect_builtin_call(m, v, "record").ok_or_else(|| refuse(v, m))?;
    if !vrec.args.is_empty() {
        return Err(refuse(v, m));
    }

    let mut components = Vec::with_capacity(rec.named.len());
    for field in rec.named.iter() {
        if field.kind != NamedKind::Field {
            return Err(refuse(measure_expr, m));
        }
        // The matching value field, by name.
        let pinned = lookup_field(m, &vrec.named, field.name).ok_or_else(|| refuse(v, m))?;

        // The component value is either `(%ref self x)` → draw binding, or an
        // inline `draw(mᵢ)`. Anything else (a derived expression, a bare
        // constant, a nested combinator) → refuse.
        let (measure, draw_binding) =
            resolve_component_draw(m, field.value).ok_or_else(|| refuse(field.value, m))?;
        components.push(Component {
            measure,
            pinned,
            draw_binding,
        });
    }

    if components.is_empty() {
        return Err(refuse(measure_expr, m));
    }
    Ok(components)
}

/// Resolve a record-field value to its underlying `draw(mᵢ)`, returning the
/// component measure `mᵢ` and (when reached through a self-reference) the draw
/// binding. Returns `None` for any shape that is not a *direct* draw — including
/// a binding whose RHS is a derived expression (audit H1/H3: refuse, never
/// re-materialise a draw through arithmetic).
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

/// If `id` is `draw(mᵢ)`, return `mᵢ`; otherwise `None`.
fn draw_argument(m: &Module, id: NodeId) -> Option<NodeId> {
    let c = expect_builtin_call(m, id, "draw")?;
    if c.args.len() != 1 {
        return None;
    }
    Some(c.args[0])
}

/// Build `builtin_logdensityof(kernel, kernel_input, pinned)` for one component
/// measure. The component must be a builtin distribution constructor applied to
/// keyword arguments — `Normal(mu = .., sigma = ..)` — which we split into the
/// bare kernel-constructor symbol (the `kernel` arg) and a `record(..)` of the
/// keyword parameters (the `kernel_input` arg), matching the `builtin_logdensityof`
/// call shape (`infer/tests/measure_eval_prims.rs`). Other constructor shapes
/// (positional args, module-qualified, `locscale`-wrapped, …) are refused.
fn build_density_term(
    m: &mut Module,
    measure: NodeId,
    pinned: NodeId,
) -> Result<NodeId, RefuseError> {
    // Read the constructor shape before any mutation.
    let (ctor_sym, kwargs): (Symbol, Vec<(Symbol, NodeId)>) = {
        let Node::Call(c) = m.node(measure) else {
            return Err(refuse(measure, m));
        };
        let CallHead::Builtin(sym) = c.head else {
            // A user / module-qualified constructor — out of scope for now.
            return Err(refuse(measure, m));
        };
        // The constructor must take only keyword parameters (no positional args),
        // which become the kernel_input record's fields.
        if !c.args.is_empty() {
            return Err(refuse(measure, m));
        }
        let mut kwargs = Vec::with_capacity(c.named.len());
        for n in c.named.iter() {
            if n.kind != NamedKind::Kwarg {
                return Err(refuse(measure, m));
            }
            kwargs.push((n.name, n.value));
        }
        (sym, kwargs)
    };

    // kernel arg: the bare constructor as a value (`(%const Normal)`), matching
    // the dump shape `(builtin_logdensityof Normal (record ..) ..)`.
    let kernel = m.alloc(Node::Const(ctor_sym));

    // kernel_input: `record(%field <param> <value> ...)` from the kwargs.
    let kernel_input = build_record(m, &kwargs);

    let builtin = m.intern("builtin_logdensityof");
    Ok(m.alloc(Node::Call(Call {
        head: CallHead::Builtin(builtin),
        args: vec![kernel, kernel_input, pinned].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    })))
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
/// fold left into nested binary `add(acc, term)` calls (the §06 Σ).
fn fold_add(m: &mut Module, terms: &[NodeId]) -> NodeId {
    debug_assert!(!terms.is_empty(), "fold_add requires at least one term");
    let mut acc = terms[0];
    let add = m.intern("add");
    for &t in &terms[1..] {
        acc = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(add),
            args: vec![acc, t].into(),
            named: Vec::<NamedArg>::new().into(),
            inputs: None,
        }));
    }
    acc
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

/// A refusal naming the construct at `id` — the determiniser reports, never
/// mis-lowers, anything outside the independent-`record`-of-`draw`s case.
fn refuse(id: NodeId, m: &Module) -> RefuseError {
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
        reason: "density lowering supports only an independent `record` of direct `draw`s \
                 (joint/iid/marginal/derived cases are later tasks)"
            .to_string(),
    }
}
