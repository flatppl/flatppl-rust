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
//!   * record-form — components are single-named-variate draws; output is a
//!     merged record; kernel inputs bind by field name (auto-splat).
//!   * scalar-cat — components are scalar draws; output is a vector; a kernel's
//!     single input binds to the `cat` of all prior slices.

use crate::density::{
    MEASURE_COMBINATOR_OPS, build_call, build_density_term, builtin_name, draw_argument,
    expect_builtin_call, fold_add, resolve_ref_one,
};
use crate::kernel::{Kernel, resolve_kernel, substitute_ref};
use crate::refuse::RefuseError;
use flatppl_core::{Module, NamedKind, Node, NodeId, Scalar, Symbol};

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

    // Resolve the base (component 0); its record-field presence picks the family.
    let base = resolve_base(m, args[0])
        .ok_or_else(|| refuse_jc(node, "jointchain base is not a single-draw measure"))?;
    match base.field {
        Some(_) => lower_record_family(m, node, &args, base, v),
        None => lower_scalar_family(m, node, &args, base, v),
    }
}

/// Record-form: components are single-field-record draws; the point is a record
/// literal; kernel inputs bind to prior fields by name.
fn lower_record_family(
    m: &mut Module,
    node: NodeId,
    args: &[NodeId],
    base: Component,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    // The point must be a record literal `record(%field f val …)`.
    let (v_resolved, _) = resolve_ref_one(m, v);
    let point: Vec<(Symbol, NodeId)> = {
        let rec = expect_builtin_call(m, v_resolved, "record")
            .ok_or_else(|| refuse_jc(node, "record-form jointchain point must be a record"))?;
        if !rec.args.is_empty() {
            return Err(refuse_jc(node, "record point with positional args"));
        }
        rec.named.iter().map(|n| (n.name, n.value)).collect()
    };
    let field_value = |f: Symbol| -> Option<NodeId> {
        point.iter().find(|(nm, _)| *nm == f).map(|(_, val)| *val)
    };

    // env: field name -> its realized value node, accumulated left to right.
    let mut env: Vec<(Symbol, NodeId)> = Vec::with_capacity(args.len());
    let mut terms: Vec<NodeId> = Vec::with_capacity(args.len());

    // Component 0 (base): no inputs.
    let f0 = base.field.expect("record family base has a field");
    let s0 = field_value(f0)
        .ok_or_else(|| refuse_jc(node, "point is missing the base variate field"))?;
    terms.push(build_density_term(m, base.dist, s0)?);
    env.push((f0, s0));

    // Components 1..n (kernels).
    for &k_arg in &args[1..] {
        let kernel = resolve_kernel(m, k_arg)
            .ok_or_else(|| refuse_jc(node, "jointchain kernel is not a kernelof(...)"))?;
        let comp = resolve_kernel_component(m, &kernel)
            .ok_or_else(|| refuse_jc(node, "kernel body is not a single-field-record draw"))?;
        let fi = comp.field.ok_or_else(|| {
            refuse_jc(
                node,
                "record-form jointchain kernel body is not a single-field-record draw",
            )
        })?;

        // Bind each input: match its NAME to a prior field, substitute its REF
        // SYMBOL in the constructor with that field's realized value.
        let mut dist_i = comp.dist;
        for (name, target) in &kernel.inputs {
            let value = env
                .iter()
                .find(|(nm, _)| nm == name)
                .map(|(_, val)| *val)
                .ok_or_else(|| refuse_jc(node, "kernel input names a non-prior variate field"))?;
            dist_i = substitute_ref(m, dist_i, target.name, value);
        }

        let si = field_value(fi)
            .ok_or_else(|| refuse_jc(node, "point is missing a kernel variate field"))?;
        terms.push(build_density_term(m, dist_i, si)?);
        env.push((fi, si));
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

/// Resolve a kernel body to its single-draw `Component` (no `lawof`; the inputs
/// come from the `Kernel`).
fn resolve_kernel_component(m: &Module, kernel: &Kernel) -> Option<Component> {
    let (field, dist) = resolve_single_draw(m, kernel.body)?;
    Some(Component { field, dist })
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
        let kernel = resolve_kernel(m, k_arg)
            .ok_or_else(|| refuse_jc(node, "jointchain kernel is not a kernelof(...)"))?;
        // Scalar-cat kernels take exactly one input: the cat of prior variates.
        if kernel.inputs.len() != 1 {
            return Err(refuse_jc(
                node,
                "scalar-cat kernel must take exactly one input (the cat of priors)",
            ));
        }
        let comp = resolve_kernel_component(m, &kernel)
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
        let target = kernel.inputs[0].1.name;
        let dist_i = substitute_ref(m, comp.dist, target, cat);

        let idx = m.alloc(Node::Lit(Scalar::Int((i + 1) as i64)));
        let si = build_call(m, "get0", &[v, idx]);
        terms.push(build_density_term(m, dist_i, si)?);
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
            resolve_kernel(m, resolved)
                .and_then(|k| resolve_single_draw(m, k.body))
                .map(|(_, dist)| {
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
