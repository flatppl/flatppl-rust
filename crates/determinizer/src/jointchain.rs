//! `jointchain(M, K₁, …, Kₙ)` density lowering (spec §06 "Dependent
//! composition"). Unlike `kchain` (which marginalizes intermediate latents —
//! see `marginal.rs`), `jointchain` KEEPS all variates, so its density is the
//! product of the components' conditional densities — no integral:
//!
//! ```text
//! logdensityof(jointchain(M, K₁, …), point)
//!   = logdensityof(M, s₀) + Σᵢ logdensityof(Kᵢ(prior variates), sᵢ)
//! ```
//!
//! Each kernel's boundary inputs bind to the realized variates of the
//! components to its left (spec: `b ~ K(a)`). This module recognizes two
//! variate families and refuses everything else (refuse-don't-mislower):
//!   * record-form — component record fields merge; ordinary application passes
//!     the record of prior fields to the next kernel.
//!   * scalar-cat — components are scalar draws; output is a vector; a kernel's
//!     single input binds to the `cat` of all prior slices.

use crate::density::{
    MEASURE_COMBINATOR_OPS, build_call, build_density_term, build_record, builtin_name,
    draw_argument, expect_builtin_call, fold_add, lookup_field, lower_measure_density,
    materialize_record_value, resolve_ref, resolve_ref_chain,
};
use crate::kernel::resolve_reified;
use crate::refuse::RefuseError;
use flatppl_core::{Call, CallHead, Module, NamedKind, Node, NodeId, Scalar, Symbol, Type};

/// A resolved single-draw component: its variate field name (record-form) or
/// `None` (scalar-cat), and its distribution constructor. Kernel inputs are
/// read from the `Kernel` directly (not carried here).
struct Component {
    field: Option<Symbol>,
    dist: NodeId,
}

/// Lower `logdensityof(jointchain(C₀, …, Cₙ), v)`.
pub(crate) fn lower_jointchain(
    m: &mut Module,
    node: NodeId,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    if let Some(components) = keyword_components(m, node) {
        return lower_keyword_family(m, node, &components, v);
    }
    let args: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "jointchain")
            .ok_or_else(|| refuse_jc(node, "expected jointchain"))?;
        if !c.named.is_empty() {
            return Err(refuse_jc(
                node,
                "a jointchain with both positional and keyword components is not lowered",
            ));
        }
        if c.args.len() < 2 {
            return Err(refuse_jc(
                node,
                "jointchain needs a base and at least one kernel",
            ));
        }
        c.args.to_vec()
    };

    let (base_node, _) = resolve_ref_chain(m, args[0]);
    if matches!(m.type_of(base_node), Some(Type::Measure { .. }))
        && record_fields(m, args[0]).is_some()
    {
        return lower_record_family(m, node, &args, v);
    }

    let base = resolve_base(m, args[0])
        .ok_or_else(|| refuse_jc(node, "jointchain base is not a single-draw measure"))?;
    lower_scalar_family(m, node, &args, base, v)
}

/// The ordered variate field names of a RECORD-FAMILY `jointchain` `node` — the
/// base variate's field first, then each kernel's field, in chain order. `None`
/// if `node` is not a positional `jointchain` whose base and every kernel resolve
/// to single-field-record draws (e.g. the scalar-cat family, a keyword-form
/// jointchain, or a malformed chain) — such a chain has no field-keyed variates
/// to project by name.
///
/// Used by the determiniser's structural-projection prefix-keep guard
/// ([`crate::density`]): a projection keeping a leading PREFIX of these fields is
/// closed-form (jointchain kernels read only PRIOR variates, so the dropped
/// trailing kernels are normalized Markov kernels that integrate to 1 and drop);
/// any other keep drops a depended-upon variate — the intractable `kchain`
/// integral — and must refuse.
pub(crate) fn record_variate_fields(m: &Module, node: NodeId) -> Option<Vec<Symbol>> {
    if let Some(components) = keyword_components(m, node) {
        return (components.len() >= 2)
            .then(|| components.iter().map(|(label, _)| *label).collect());
    }
    let args: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "jointchain")?;
        if !c.named.is_empty() || c.args.len() < 2 {
            return None;
        }
        c.args.to_vec()
    };
    let base = resolve_base(m, args[0])?;
    let mut fields = Vec::with_capacity(args.len());
    fields.push(base.field?);
    for &k_arg in &args[1..] {
        fields.push(resolve_step_output(m, k_arg)?.field?);
    }
    Some(fields)
}

/// True iff the jointchain kernel argument `k_arg` resolves to a single-draw body
/// whose distribution constructor is a **confirmed normalized probability
/// measure** — its OWN inferred `Mass` is `Normalized` (a §08 distribution like
/// `Normal`/`Gamma`), NOT a reference/base measure `Lebesgue`/`Counting` (which
/// carry `LocallyFinite`/`Finite`/`Unknown` mass — the reference measures, not
/// probability measures) and NOT an un-normalized combinator (which
/// [`resolve_single_draw`] already rejects, so this returns `false` for those).
///
/// The determiniser's prefix-keep projection ([`crate::density`]) uses this to
/// decide whether a DROPPED trailing kernel integrates to 1. It must NOT trust the
/// kernel-TYPE mass: inference types EVERY `kernelof(...)` as `Mass::Normalized`
/// unconditionally, ignoring the body (`crates/infer/src/ops.rs`), so an
/// improper-body kernel would falsely pass a kernel-type mass check and be dropped
/// as if it had unit mass — silently lowering an infinite-mass marginal to a
/// finite WRONG density. Reading the BODY measure's own mass closes that hole
/// (refuse-don't-mislower).
pub(crate) fn kernel_body_is_normalized(m: &Module, k_arg: NodeId) -> bool {
    let Some(comp) = resolve_step_output(m, k_arg) else {
        return false;
    };
    matches!(
        m.type_of(comp.dist),
        Some(flatppl_core::Type::Measure {
            mass: flatppl_core::Mass::Normalized,
            ..
        })
    )
}

/// Record fields from an inferred measure domain or a reified value body.
fn record_fields(m: &Module, node: NodeId) -> Option<Vec<Symbol>> {
    let (resolved, _) = resolve_ref_chain(m, node);
    let ty = m.type_of(node).or_else(|| m.type_of(resolved))?;
    let domain = match ty {
        Type::Measure { domain, .. } => domain.as_ref(),
        ty => ty,
    };
    match domain {
        Type::Record(fields) => Some(fields.iter().map(|(name, _)| *name).collect()),
        _ => None,
    }
}

/// The components of `joint(K1, K2, ...)` / `joint(n1 = K1, ...)` when `node` is such a joint
/// kernel (spec §06 "Joint measures": a kernel that fans its input out to every component),
/// as `(label, component)` pairs, the label being `None` in the positional form.
fn joint_kernel(m: &Module, node: NodeId) -> Option<Vec<(Option<Symbol>, NodeId)>> {
    let (resolved, _) = resolve_ref_chain(m, node);
    if !matches!(m.type_of(resolved), Some(Type::Kernel { .. })) {
        return None;
    }
    let call = expect_builtin_call(m, resolved, "joint")?;
    let mut components: Vec<_> = call.args.iter().map(|&arg| (None, arg)).collect();
    components.extend(call.named.iter().map(|arg| (Some(arg.name), arg.value)));
    (call.args.is_empty() || call.named.is_empty()).then_some(components)
}

/// The variate fields of a joint kernel: the labels of the keyword form, or the fields of
/// every component, in order, for the positional form. `None` unless every component's
/// variate is a record.
fn joint_kernel_fields(m: &Module, node: NodeId) -> Option<Vec<Symbol>> {
    let components = joint_kernel(m, node)?;
    if components.iter().all(|(label, _)| label.is_some()) {
        return Some(components.iter().filter_map(|(label, _)| *label).collect());
    }
    let mut fields = Vec::new();
    for (_, component) in components {
        let own = match resolve_reified(m, component) {
            Some(kernel) => record_fields(m, kernel.body),
            None => joint_kernel_fields(m, component).or_else(|| record_fields(m, component)),
        }?;
        fields.extend(own);
    }
    Some(fields)
}

/// Score each record component through the ordinary measure dispatcher. Named
/// applications retain kernel conditioning and simultaneous boundary substitution.
fn lower_record_family(
    m: &mut Module,
    node: NodeId,
    args: &[NodeId],
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    let value = materialize_record_value(m, v);
    let point = expect_builtin_call(m, value, "record")
        .filter(|call| call.args.is_empty())
        .ok_or_else(|| refuse_jc(node, "record-form jointchain point must be a record"))?
        .named
        .to_vec();
    let mut env: Vec<(Symbol, NodeId)> = Vec::new();
    let mut terms = Vec::with_capacity(args.len());
    for (i, &component) in args.iter().enumerate() {
        let (measure, fields) = if i == 0 {
            (component, record_fields(m, component))
        } else {
            let (resolved, _) = resolve_ref_chain(m, component);
            if !matches!(m.type_of(resolved), Some(Type::Kernel { .. })) {
                return Err(refuse_jc(
                    node,
                    "jointchain component is not a transition kernel",
                ));
            }
            let fields = match resolve_reified(m, component) {
                Some(kernel) => record_fields(m, kernel.body),
                None if joint_kernel(m, component).is_some() => joint_kernel_fields(m, component),
                None => {
                    return Err(refuse_jc(
                        node,
                        "jointchain kernel is not a reified callable",
                    ));
                }
            };
            let prior = build_record(m, &env);
            let applied = m.alloc(Node::Call(Call {
                head: CallHead::User(component),
                args: Box::new([prior]),
                named: Box::default(),
                inputs: None,
            }));
            (applied, fields)
        };
        let fields = fields.ok_or_else(|| {
            refuse_jc(
                node,
                "record-form jointchain component domain is not a record",
            )
        })?;
        let mut values = Vec::with_capacity(fields.len());
        for field in fields {
            if env.iter().any(|(name, _)| *name == field) {
                return Err(refuse_jc(
                    node,
                    "jointchain record components have duplicate fields",
                ));
            }
            let value = lookup_field(m, &point, field)
                .ok_or_else(|| refuse_jc(node, "point is missing a component variate field"))?;
            values.push((field, value));
        }
        let subrecord = build_record(m, &values);
        terms.push(lower_measure_density(m, measure, subrecord)?);
        env.extend(values);
    }

    Ok(fold_add(m, &terms))
}

/// The `(label, component)` pairs of a keyword-form `jointchain(name1 = M, name2 = K1, ...)`,
/// `None` for any other call.
pub(crate) fn keyword_components(m: &Module, node: NodeId) -> Option<Vec<(Symbol, NodeId)>> {
    let c = expect_builtin_call(m, node, "jointchain")?;
    if !c.args.is_empty() || c.named.is_empty() {
        return None;
    }
    Some(c.named.iter().map(|arg| (arg.name, arg.value)).collect())
}

/// The chain over the first `k` components of the jointchain `node`: the sub-chain whose
/// density is the marginal of a dependency-respecting prefix.
pub(crate) fn prefix_chain(m: &mut Module, node: NodeId, k: usize) -> Option<NodeId> {
    let call = expect_builtin_call(m, node, "jointchain")?.clone();
    let sym = m.intern("jointchain");
    Some(m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: call.args.iter().copied().take(k).collect(),
        named: call.named.iter().copied().take(k).collect(),
        inputs: None,
    })))
}

/// Keyword form (spec §06 "Keyword form"): the variate is the record of the labelled
/// component variates, and "each later kernel receives the record of the preceding
/// named variates by the ordinary calling convention, so `jointchain(a = M, b = K)`
/// calls `K(record(a = a))`". The density is the product of the conditionals: each
/// component scored at its label's field of the point, a kernel applied to the record
/// of the preceding fields.
fn lower_keyword_family(
    m: &mut Module,
    node: NodeId,
    components: &[(Symbol, NodeId)],
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    let value = materialize_record_value(m, v);
    let point = expect_builtin_call(m, value, "record")
        .filter(|call| call.args.is_empty())
        .ok_or_else(|| refuse_jc(node, "keyword-form jointchain point must be a record"))?
        .named
        .to_vec();
    let mut env: Vec<(Symbol, NodeId)> = Vec::new();
    let mut terms = Vec::with_capacity(components.len());
    for (i, &(label, component)) in components.iter().enumerate() {
        if env.iter().any(|(name, _)| *name == label) {
            return Err(refuse_jc(node, "jointchain labels are not distinct"));
        }
        let variate = lookup_field(m, &point, label)
            .ok_or_else(|| refuse_jc(node, "point is missing a component's labelled variate"))?;
        let measure = if i == 0 {
            component
        } else {
            let (resolved, _) = resolve_ref_chain(m, component);
            if !matches!(m.type_of(resolved), Some(Type::Kernel { .. })) {
                return Err(refuse_jc(
                    node,
                    "jointchain component is not a transition kernel",
                ));
            }
            if resolve_reified(m, component).is_none() && joint_kernel(m, component).is_none() {
                return Err(refuse_jc(
                    node,
                    "jointchain kernel is not a reified callable",
                ));
            }
            // The kernel receives the record of the preceding labelled variates by the
            // ordinary call; the engine decides between splat and whole.
            let prior = build_record(m, &env);
            m.alloc(Node::Call(Call {
                head: CallHead::User(component),
                args: Box::new([prior]),
                named: Box::default(),
                inputs: None,
            }))
        };
        terms.push(lower_measure_density(m, measure, variate)?);
        env.push((label, variate));
    }
    Ok(fold_add(m, &terms))
}

/// Resolve the jointchain base `args[0]` to a `Component`. Accepts
/// `lawof(record(f = <draw>))` (record family) or `lawof(<draw>)` — a bare
/// scalar draw with no record wrapper (scalar family).
fn resolve_base(m: &Module, base_arg: NodeId) -> Option<Component> {
    let (resolved, _) = resolve_ref(m, base_arg);
    let inner = match expect_builtin_call(m, resolved, "lawof") {
        Some(law) if law.args.len() == 1 => resolve_ref(m, law.args[0]).0,
        Some(_) => return None,
        None => resolved,
    };
    resolve_single_draw(m, inner).map(|(field, dist)| Component { field, dist })
}

/// Resolve a chain step to the single-draw `Component` its output measure is,
/// looking through an adapter. The step is a reified callable (`kernelof` or
/// `functionof`). An adapter such as `pars -> k(a = pars.a)` (spec §04, lambda
/// sugar) reifies a call of another kernel, whose output measure is that
/// kernel's, with its inputs bound to other arguments — so the field name and
/// the body's mass do not depend on the arguments and are read from `k`. Only
/// those two facts are asked of the result; a caller that needs the step's
/// inputs reads them from the step itself ([`resolve_reified`]).
fn resolve_step_output(m: &Module, k_arg: NodeId) -> Option<Component> {
    let mut step = k_arg;
    for _ in 0..8 {
        let kernel = resolve_reified(m, step)?;
        let (body, _) = resolve_ref(m, kernel.body);
        match m.node(body) {
            Node::Call(call) => match call.head {
                CallHead::User(callee) => step = callee,
                CallHead::Builtin(_) => {
                    return resolve_single_draw(m, kernel.body)
                        .map(|(field, dist)| Component { field, dist });
                }
            },
            _ => {
                return resolve_single_draw(m, kernel.body)
                    .map(|(field, dist)| Component { field, dist });
            }
        }
    }
    None
}

/// Peel an optional single-field `record(f = X)` wrapper, then an optional
/// `draw(dist)`, to a builtin distribution constructor. Returns
/// `(Some(field), dist)` for the record form, `(None, dist)` for a bare scalar
/// draw. `None` for a multi-field record, positional record, non-constructor,
/// or a measure-COMBINATOR call (e.g. `superpose(...)`) — a combinator is a
/// builtin call too, but it is not a single primitive draw, so treating it as
/// one here would let a `resolve_base`/`resolve_kernel_component` caller
/// silently mis-lower it as a leaf distribution instead of refusing (and,
/// worse, a downstream refusal would then name the inner combinator instead
/// of `jointchain`).
fn resolve_single_draw(m: &Module, expr: NodeId) -> Option<(Option<Symbol>, NodeId)> {
    let (resolved, _) = resolve_ref(m, expr);
    let (inner, field) = if let Some(rec) = expect_builtin_call(m, resolved, "record") {
        if !rec.args.is_empty() || rec.named.len() != 1 || rec.named[0].kind != NamedKind::Field {
            return None;
        }
        (
            resolve_ref(m, rec.named[0].value).0,
            Some(rec.named[0].name),
        )
    } else {
        (resolved, None)
    };
    let dist = match draw_argument(m, inner) {
        Some(d) => resolve_ref(m, d).0,
        None => inner,
    };
    let name = builtin_name(m, dist)?;
    if MEASURE_COMBINATOR_OPS.contains(&name) {
        return None;
    }
    Some((field, dist))
}

/// Scalar-cat: components are scalar draws; the point is a vector; slice i via
/// `get0(v, i)`. A kernel has exactly ONE input, bound to the `cat` of all
/// prior slices — the scalar `s₀` for one prior, `vector(s₀,…,s_{i-1})` for more.
fn lower_scalar_family(
    m: &mut Module,
    node: NodeId,
    args: &[NodeId],
    base: Component,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    // Fail-closed: every component's variate must be CONFIRMED scalar, else a
    // get0 slice would silently drop/misplace slots (cf. lower_joint's guard).
    for &comp_arg in args {
        if !component_is_scalar(m, comp_arg) {
            return Err(refuse_jc(
                node,
                "scalar-cat component variate is not confirmed scalar",
            ));
        }
    }

    let mut env: Vec<NodeId> = Vec::with_capacity(args.len());
    let mut terms: Vec<NodeId> = Vec::with_capacity(args.len());

    // Base = slice 0.
    let idx0 = m.alloc(Node::Lit(Scalar::Int(0)));
    let s0 = build_call(m, "get0", &[v, idx0]);
    terms.push(build_density_term(m, base.dist, s0)?);
    env.push(s0);

    for (i, &k_arg) in args[1..].iter().enumerate() {
        let kernel = resolve_reified(m, k_arg)
            .ok_or_else(|| refuse_jc(node, "jointchain kernel is not a reified callable"))?;
        // Scalar-cat kernels take exactly one input: the cat of prior variates.
        if kernel.inputs.len() != 1 {
            return Err(refuse_jc(
                node,
                "scalar-cat kernel must take exactly one input (the cat of priors)",
            ));
        }
        let comp = resolve_step_output(m, k_arg)
            .ok_or_else(|| refuse_jc(node, "kernel body is not a single scalar draw"))?;
        if comp.field.is_some() {
            return Err(refuse_jc(
                node,
                "mixed families: record-form kernel in a scalar-cat chain",
            ));
        }

        // Bind the single input to cat(env): the scalar itself for one prior,
        // else a vector of all prior slices.
        let cat = if env.len() == 1 {
            env[0]
        } else {
            build_call(m, "vector", &env)
        };
        let applied = m.alloc(Node::Call(Call {
            head: CallHead::User(k_arg),
            args: Box::new([cat]),
            named: Box::default(),
            inputs: None,
        }));

        let idx = m.alloc(Node::Lit(Scalar::Int((i + 1) as i64)));
        let si = build_call(m, "get0", &[v, idx]);
        terms.push(lower_measure_density(m, applied, si)?);
        env.push(si);
    }

    Ok(fold_add(m, &terms))
}

/// Is the component argument (a base measure or a `kernelof`) confirmed to have
/// a SCALAR variate? Reads the inferred measure/kernel domain; fail-closed on
/// unknown/deferred (returns false → the caller refuses).
fn component_is_scalar(m: &Module, comp_arg: NodeId) -> bool {
    let (resolved, _) = resolve_ref(m, comp_arg);
    match m.type_of(resolved) {
        Some(flatppl_core::Type::Measure { domain, .. }) => {
            matches!(domain.as_ref(), flatppl_core::Type::Scalar(_))
        }
        Some(flatppl_core::Type::Kernel { .. }) => {
            // A kernel's variate is its body's domain; resolve the body's draw
            // constructor and check its measure domain.
            resolve_step_output(m, resolved)
                .map(|Component { dist, .. }| {
                    matches!(
                        m.type_of(dist),
                        Some(flatppl_core::Type::Measure { domain, .. })
                            if matches!(domain.as_ref(), flatppl_core::Type::Scalar(_))
                    )
                })
                .unwrap_or(false)
        }
        _ => false,
    }
}

fn refuse_jc(node: NodeId, reason: &str) -> RefuseError {
    RefuseError {
        node,
        construct: "jointchain".to_string(),
        reason: reason.to_string(),
    }
}

/// Sample `jointchain(C₀, K₁, …)` in measure position (spec §06): draw the base,
/// then each kernel applied to the variates to its left, threading the rng from
/// component to component, exactly as §06's equivalence with stochastic nodes reads
/// (`a ~ M1; b ~ K2(a); c ~ K3([a, b])`). Every component is sampled by the ordinary
/// closed-measure dispatcher, and each sampled variate is bound once, so later kernels
/// and the output read one draw by name.
///
/// The three variate families and their refusals mirror [`lower_jointchain`]:
///   * keyword — each kernel receives the record of the preceding labelled variates;
///     the variate is that record.
///   * record (positional, record variates) — each kernel receives the record of the
///     preceding fields; the variate is the merged record (§06 `cat` of records).
///   * scalar-cat (positional, scalar variates) — a kernel's one input is the prior
///     scalar, or the vector of all prior scalars; the variate is their `cat`.
pub(crate) fn lower_jointchain_sample(
    m: &mut Module,
    node: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    if let Some(components) = keyword_components(m, node) {
        return sample_keyword_family(m, node, &components, rng);
    }
    let args: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "jointchain")
            .ok_or_else(|| refuse_jc(node, "expected jointchain"))?;
        if !c.named.is_empty() {
            return Err(refuse_jc(
                node,
                "a jointchain with both positional and keyword components is not lowered",
            ));
        }
        if c.args.len() < 2 {
            return Err(refuse_jc(
                node,
                "jointchain needs a base and at least one kernel",
            ));
        }
        c.args.to_vec()
    };
    let (base_node, _) = resolve_ref_chain(m, args[0]);
    if matches!(m.type_of(base_node), Some(Type::Measure { .. }))
        && record_fields(m, args[0]).is_some()
    {
        return sample_record_family(m, node, &args, rng);
    }
    sample_scalar_family(m, node, &args, rng)
}

/// `K(prior)` for a chain step, after the same kernel check the density side makes.
fn apply_step(
    m: &mut Module,
    node: NodeId,
    component: NodeId,
    prior: NodeId,
) -> Result<NodeId, RefuseError> {
    let (resolved, _) = resolve_ref_chain(m, component);
    if !matches!(m.type_of(resolved), Some(Type::Kernel { .. })) {
        return Err(refuse_jc(
            node,
            "jointchain component is not a transition kernel",
        ));
    }
    if resolve_reified(m, component).is_none() && joint_kernel(m, component).is_none() {
        return Err(refuse_jc(
            node,
            "jointchain kernel is not a reified callable",
        ));
    }
    Ok(m.alloc(Node::Call(Call {
        head: CallHead::User(component),
        args: Box::new([prior]),
        named: Box::default(),
        inputs: None,
    })))
}

/// Sample one component and bind its variate to a fresh synthetic binding, so every
/// later reader shares the one draw.
fn sample_component(
    m: &mut Module,
    measure: NodeId,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let (value, next) = crate::sample::lower_closed_measure_sample(m, measure, rng)?;
    Ok((bind_draw(m, value), next))
}

/// Bind a draw to a fresh synthetic binding and return a reference to it.
fn bind_draw(m: &mut Module, value: NodeId) -> NodeId {
    let mut index = 0usize;
    let name = loop {
        let name = m.intern(&format!("__jointchain_{index}"));
        if m.binding_by_name(name).is_none() {
            break name;
        }
        index += 1;
    };
    m.add_binding(flatppl_core::Binding {
        name,
        rhs: value,
        doc: None,
        public: false,
        synthetic: true,
    });
    m.alloc(Node::Ref(flatppl_core::Ref {
        ns: flatppl_core::RefNs::SelfMod,
        name,
    }))
}

fn sample_keyword_family(
    m: &mut Module,
    node: NodeId,
    components: &[(Symbol, NodeId)],
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let mut env: Vec<(Symbol, NodeId)> = Vec::with_capacity(components.len());
    let mut cur = rng;
    for (i, &(label, component)) in components.iter().enumerate() {
        if env.iter().any(|(name, _)| *name == label) {
            return Err(refuse_jc(node, "jointchain labels are not distinct"));
        }
        let measure = if i == 0 {
            component
        } else {
            let prior = build_record(m, &env);
            apply_step(m, node, component, prior)?
        };
        let (value, next) = sample_component(m, measure, cur)?;
        env.push((label, value));
        cur = next;
    }
    Ok((build_record(m, &env), cur))
}

fn sample_record_family(
    m: &mut Module,
    node: NodeId,
    args: &[NodeId],
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let mut env: Vec<(Symbol, NodeId)> = Vec::new();
    let mut cur = rng;
    for (i, &component) in args.iter().enumerate() {
        let (measure, fields) = if i == 0 {
            (component, record_fields(m, component))
        } else {
            let fields = match resolve_reified(m, component) {
                Some(kernel) => record_fields(m, kernel.body),
                None => joint_kernel_fields(m, component),
            };
            let prior = build_record(m, &env);
            (apply_step(m, node, component, prior)?, fields)
        };
        let fields = fields.ok_or_else(|| {
            refuse_jc(
                node,
                "record-form jointchain component domain is not a record",
            )
        })?;
        let (value, next) = sample_component(m, measure, cur)?;
        for field in fields {
            if env.iter().any(|(name, _)| *name == field) {
                return Err(refuse_jc(
                    node,
                    "jointchain record components have duplicate fields",
                ));
            }
            let name: Box<str> = m.resolve(field).into();
            let key = m.alloc(Node::Lit(Scalar::Str(name)));
            env.push((field, build_call(m, "get", &[value, key])));
        }
        cur = next;
    }
    Ok((build_record(m, &env), cur))
}

fn sample_scalar_family(
    m: &mut Module,
    node: NodeId,
    args: &[NodeId],
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    for &component in args {
        if !component_is_scalar(m, component) {
            return Err(refuse_jc(
                node,
                "scalar-cat component variate is not confirmed scalar",
            ));
        }
    }
    let mut env: Vec<NodeId> = Vec::with_capacity(args.len());
    let mut cur = rng;
    for (i, &component) in args.iter().enumerate() {
        let measure = if i == 0 {
            component
        } else {
            let kernel = resolve_reified(m, component)
                .ok_or_else(|| refuse_jc(node, "jointchain kernel is not a reified callable"))?;
            if kernel.inputs.len() != 1 {
                return Err(refuse_jc(
                    node,
                    "scalar-cat kernel must take exactly one input (the cat of priors)",
                ));
            }
            let prior = if env.len() == 1 {
                env[0]
            } else {
                build_call(m, "vector", &env)
            };
            apply_step(m, node, component, prior)?
        };
        let (value, next) = sample_component(m, measure, cur)?;
        env.push(value);
        cur = next;
    }
    Ok((build_call(m, "cat", &env), cur))
}

/// `iid(jointchain(M, K₁, …), n)`: `n` independent chains, sampled one component at
/// a time across all lanes. §06 gives `iid(J, n)` the product `J^⊗n`, and a lane's
/// `J` integrates `K(a, ·)` over `M`, so the product is `jointchain(iid(M, n), a⃗ ->
/// ⊗ᵢ K(aᵢ))`: the base drawn once per lane, then each kernel applied lane by lane to
/// the variates to its left. Each component is one batched draw over the lanes, and
/// the key advances once per component field.
///
/// Every component must be a scalar law per lane, or a keyword `joint` of scalar laws
/// in the record form. The variate is an `n`-row table for the record and keyword
/// forms, and an `n × k` matrix for `k` scalar components (§06 `iid`).
pub(crate) fn lower_iid_jointchain_sample(
    m: &mut Module,
    chain: NodeId,
    n: usize,
    rng: NodeId,
) -> Result<(NodeId, NodeId), RefuseError> {
    let (components, record) = match keyword_components(m, chain) {
        Some(components) => (
            components
                .into_iter()
                .map(|(label, component)| (Some(label), component))
                .collect::<Vec<_>>(),
            true,
        ),
        None => {
            let c = expect_builtin_call(m, chain, "jointchain")
                .ok_or_else(|| refuse_jc(chain, "expected jointchain"))?;
            if !c.named.is_empty() || c.args.len() < 2 {
                return Err(refuse_jc(
                    chain,
                    "iid of a jointchain needs a base and kernels, all positional or all named",
                ));
            }
            let args: Vec<_> = c.args.iter().map(|&arg| (None, arg)).collect();
            let record = record_fields(m, args[0].1).is_some();
            (args, record)
        }
    };
    let mut columns: Vec<(Option<Symbol>, NodeId)> = Vec::new();
    let mut cur = rng;
    for (i, &(label, component)) in components.iter().enumerate() {
        if i == 0 {
            for (field, law) in lane_laws(m, chain, label, record, component)? {
                let size = m.alloc(Node::Lit(Scalar::Int(n as i64)));
                let iid = build_call(m, "iid", &[law, size]);
                let (value, next) = crate::sample::lower_closed_measure_sample(m, iid, cur)?;
                columns.push((field, lane_column(m, chain, law, value, n)?));
                cur = next;
            }
            continue;
        }
        let sources: Vec<NodeId> = columns.iter().map(|&(_, column)| column).collect();
        let lanes = crate::sample::Lanes::over(m, &sources)?;
        let prior = if record {
            let fields: Vec<_> = columns
                .iter()
                .zip(&lanes.points)
                .map(|(&(field, _), &point)| (field.expect("a record lane names its field"), point))
                .collect();
            build_record(m, &fields)
        } else if let [point] = lanes.points[..] {
            point
        } else {
            build_call(m, "vector", &lanes.points)
        };
        let measure = apply_step(m, chain, component, prior)?;
        for (field, law) in lane_laws(m, chain, label, record, measure)? {
            let (value, next) = lanes.sample(m, law, cur)?;
            columns.push((field, lane_column(m, chain, law, value, n)?));
            cur = next;
        }
    }
    let value = if record {
        let mut fields = Vec::with_capacity(columns.len());
        for (field, column) in columns {
            let name = field.expect("a record lane names its field");
            if fields
                .iter()
                .any(|f: &flatppl_core::NamedArg| f.name == name)
            {
                return Err(refuse_jc(
                    chain,
                    "jointchain record components have duplicate fields",
                ));
            }
            fields.push(flatppl_core::NamedArg {
                kind: NamedKind::Field,
                name,
                value: column,
            });
        }
        let head = CallHead::Builtin(m.intern("table"));
        m.alloc(Node::Call(Call {
            head,
            args: Box::default(),
            named: fields.into(),
            inputs: None,
        }))
    } else {
        let cells: Vec<NodeId> = columns.into_iter().map(|(_, column)| column).collect();
        let vectors = build_call(m, "vector", &cells);
        build_call(m, "colstack", &[vectors])
    };
    Ok((value, cur))
}

/// The scalar lane laws of one chain component: the component itself (labelled in
/// the keyword form), or the fields of a keyword `joint` in the record form.
fn lane_laws(
    m: &mut Module,
    chain: NodeId,
    label: Option<Symbol>,
    record: bool,
    measure: NodeId,
) -> Result<Vec<(Option<Symbol>, NodeId)>, RefuseError> {
    let mut measure = resolve_ref_chain(m, measure).0;
    while let Some(body) = crate::kernel::reduce_kernel_application(m, measure) {
        measure = resolve_ref_chain(m, body).0;
    }
    if label.is_some() || !record {
        return Ok(vec![(label, measure)]);
    }
    match expect_builtin_call(m, measure, "joint") {
        Some(c) if c.args.is_empty() && !c.named.is_empty() => {
            Ok(c.named.iter().map(|f| (Some(f.name), f.value)).collect())
        }
        _ => Err(refuse_jc(
            chain,
            "iid of a record jointchain samples a keyword joint of scalar laws per lane",
        )),
    }
}

/// Bind one component's lane draws, an `n`-vector, so every later reader shares the
/// draw, typed so the next component can broadcast over its cells.
fn lane_column(
    m: &mut Module,
    chain: NodeId,
    law: NodeId,
    value: NodeId,
    n: usize,
) -> Result<NodeId, RefuseError> {
    let law = resolve_ref_chain(m, law).0;
    let ty = m
        .type_of(law)
        .cloned()
        .or_else(|| flatppl_infer::infer_expression(m, law, &[]).0.remove(&law));
    let Some(Type::Measure { domain, .. }) = ty else {
        return Err(refuse_jc(
            chain,
            "iid of a jointchain component has no law type",
        ));
    };
    if !matches!(domain.as_ref(), Type::Scalar(_)) {
        return Err(refuse_jc(
            chain,
            "iid of a jointchain samples only scalar laws per lane",
        ));
    }
    let n =
        u32::try_from(n).map_err(|_| refuse_jc(chain, "iid of a jointchain has too many lanes"))?;
    let column = bind_draw(m, value);
    m.set_type(
        column,
        Type::Array {
            shape: Box::new([flatppl_core::Dim::Static(n)]),
            elem: domain,
        },
    );
    Ok(column)
}
