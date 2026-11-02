//! Black-box interface members (flatppl-bridge).
//!
//! A `load_module` alias bound to a box ([`Module::box_binding`]) resolves its
//! members from the box's describe reply instead of from the interface
//! module's `external(anything)` declarations.

use flatppl_core::{BoxBinding, BoxKind, BoxSet, Mass, Module, Phase, Type, ValueSet};

use crate::callable::Interface;
use crate::modules::Resolved;

/// The input name of a function or kernel member whose input set is not a
/// record. A record input takes one input per field instead.
pub const BOX_INPUT: &str = "input";

/// The interface input names of a function or kernel member.
pub fn box_input_names(m: &Module, input: Option<&BoxSet>) -> Vec<String> {
    match input.map(|set| &set.ty) {
        Some(Type::Record(fields)) => fields
            .iter()
            .map(|(name, _)| m.resolve(*name).to_string())
            .collect(),
        _ => vec![BOX_INPUT.to_string()],
    }
}

/// Type member `name` of the box bound in `importer`.
pub(crate) fn resolve_box_member(
    importer: &mut Module,
    binding: &BoxBinding,
    name: &str,
) -> Result<Resolved, String> {
    let member = binding
        .members
        .get(name)
        .ok_or_else(|| format!("box `{}` has no member `{name}`", binding.name))?;
    let element = |set: &Option<BoxSet>| match set {
        Some(set) if !set.is_anything() => set.ty.clone(),
        _ => Type::Any,
    };
    let mass = if member.normalized {
        Mass::Normalized
    } else {
        Mass::Unknown
    };
    let names = box_input_names(importer, member.input.as_ref());
    let inputs: Box<[_]> = names.iter().map(|n| importer.intern(n)).collect();
    let measure = |set: &Option<BoxSet>| Type::Measure {
        domain: Box::new(element(set)),
        mass,
    };
    let (ty, result, interface) = match member.kind {
        BoxKind::Measure => (measure(&member.variate), None, None),
        BoxKind::Kernel => (
            Type::Kernel { inputs, mass },
            Some(measure(&member.variate)),
            Some(Interface::required(names, true)),
        ),
        BoxKind::Function => (
            Type::Function { inputs },
            Some(element(&member.output)),
            Some(Interface::required(names, true)),
        ),
    };
    Ok(Resolved {
        ty,
        phase: Phase::Fixed,
        vset: ValueSet::Unknown,
        set_element: None,
        result,
        catalogue: None,
        interface,
    })
}

/// `true` when `tag` names a measure member of a bound box. A measure is a
/// nullary kernel, so its primitives take no kernel-input argument.
pub(crate) fn is_box_measure(m: &Module, tag: flatppl_core::NodeId) -> bool {
    let flatppl_core::Node::Ref(flatppl_core::Ref {
        ns: flatppl_core::RefNs::Module(alias),
        name,
    }) = *m.node(tag)
    else {
        return false;
    };
    m.box_binding(alias)
        .and_then(|binding| binding.members.get(m.resolve(name)))
        .is_some_and(|member| member.kind == BoxKind::Measure)
}
