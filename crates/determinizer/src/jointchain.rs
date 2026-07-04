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
    build_density_term, draw_argument, expect_builtin_call, fold_add, refuse, resolve_ref_one,
};
use crate::kernel::{Kernel, resolve_kernel, substitute_ref};
use crate::refuse::RefuseError;
use flatppl_core::{CallHead, Module, NamedKind, Node, NodeId, Symbol};

/// A resolved single-draw component: its variate field name (record-form) or
/// `None` (scalar-cat), the distribution constructor, and its kernel inputs
/// (empty for the base measure).
struct Component {
    field: Option<Symbol>,
    dist: NodeId,
    // Read by the scalar-cat family (Task 4); record-form binds via
    // `Kernel.inputs` directly, so this field is unread here.
    #[allow(dead_code)]
    inputs: Vec<(Symbol, flatppl_core::Ref)>,
}

/// Lower `logdensityof(jointchain(C₀, …, Cₙ), v)`.
pub(crate) fn lower_jointchain(
    m: &mut Module,
    node: NodeId,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    let args: Vec<NodeId> = {
        let c = expect_builtin_call(m, node, "jointchain")
            .ok_or_else(|| refuse(node, m, "expected jointchain"))?;
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
        let fi = comp.field.expect("record family kernel has a field");

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
/// `lawof(record(f = <draw>))` (record family) or `lawof(<draw>)` /
/// `lawof(record())`-free scalar draw (scalar family).
fn resolve_base(m: &Module, base_arg: NodeId) -> Option<Component> {
    let (resolved, _) = resolve_ref_one(m, base_arg);
    let inner = match expect_builtin_call(m, resolved, "lawof") {
        Some(law) if law.args.len() == 1 => resolve_ref_one(m, law.args[0]).0,
        Some(_) => return None,
        None => resolved,
    };
    resolve_single_draw(m, inner).map(|(field, dist)| Component {
        field,
        dist,
        inputs: vec![],
    })
}

/// Resolve a kernel body to its single-draw `Component` (no `lawof`; the inputs
/// come from the `Kernel`).
fn resolve_kernel_component(m: &Module, kernel: &Kernel) -> Option<Component> {
    let (field, dist) = resolve_single_draw(m, kernel.body)?;
    Some(Component {
        field,
        dist,
        inputs: kernel.inputs.clone(),
    })
}

/// Peel an optional single-field `record(f = X)` wrapper, then an optional
/// `draw(dist)`, to a builtin distribution constructor. Returns
/// `(Some(field), dist)` for the record form, `(None, dist)` for a bare scalar
/// draw. `None` for a multi-field record, positional record, or non-constructor.
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
    if !matches!(m.node(dist), Node::Call(c) if matches!(c.head, CallHead::Builtin(_))) {
        return None;
    }
    Some((field, dist))
}

/// Scalar-cat family — implemented in Task 4. Refuse for now so record-form is
/// self-contained and independently reviewable.
fn lower_scalar_family(
    _m: &mut Module,
    node: NodeId,
    _args: &[NodeId],
    _base: Component,
    _v: NodeId,
) -> Result<NodeId, RefuseError> {
    Err(refuse_jc(
        node,
        "scalar-cat jointchain lowering is not yet implemented",
    ))
}

fn refuse_jc(node: NodeId, reason: &str) -> RefuseError {
    RefuseError {
        node,
        construct: "jointchain".to_string(),
        reason: reason.to_string(),
    }
}
