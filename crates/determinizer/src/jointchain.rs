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
    materialize_record_value, resolve_ref_chain, resolve_ref_one,
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
    let args: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "jointchain")
            .ok_or_else(|| refuse_jc(node, "expected jointchain"))?;
        if !c.named.is_empty() {
            return Err(refuse_jc(
                node,
                "keyword-form jointchain (named components) is not lowered",
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
            let kernel = resolve_reified(m, component)
                .ok_or_else(|| refuse_jc(node, "jointchain kernel is not a reified callable"))?;
            let prior = build_record(m, &env);
            let applied = m.alloc(Node::Call(Call {
                head: CallHead::User(component),
                args: Box::new([prior]),
                named: Box::default(),
                inputs: None,
            }));
            (applied, record_fields(m, kernel.body))
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

/// Resolve the jointchain base `args[0]` to a `Component`. Accepts
/// `lawof(record(f = <draw>))` (record family) or `lawof(<draw>)` — a bare
/// scalar draw with no record wrapper (scalar family).
fn resolve_base(m: &Module, base_arg: NodeId) -> Option<Component> {
    let (resolved, _) = resolve_ref_one(m, base_arg);
    let inner = match expect_builtin_call(m, resolved, "lawof") {
        Some(law) if law.args.len() == 1 => resolve_ref_one(m, law.args[0]).0,
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
        let (body, _) = resolve_ref_one(m, kernel.body);
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
    let (resolved, _) = resolve_ref_one(m, expr);
    let (inner, field) = if let Some(rec) = expect_builtin_call(m, resolved, "record") {
        if !rec.args.is_empty() || rec.named.len() != 1 || rec.named[0].kind != NamedKind::Field {
            return None;
        }
        (
            resolve_ref_one(m, rec.named[0].value).0,
            Some(rec.named[0].name),
        )
    } else {
        (resolved, None)
    };
    let dist = match draw_argument(m, inner) {
        Some(d) => resolve_ref_one(m, d).0,
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
    let (resolved, _) = resolve_ref_one(m, comp_arg);
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
