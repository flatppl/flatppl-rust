//! §06 tilt preserves the base variate and normalizes at each kernel input.

use super::*;

pub(super) fn call_type(inf: &mut Inferencer<'_, '_>, id: NodeId, args: &[ArgInfo]) -> Type {
    let [source, base] = args else {
        return Type::Deferred;
    };
    for (_, ty, _) in args {
        if !matches!(ty, Type::Kernel { .. }) && non_measure_kind(ty).is_some() {
            return failed(inf, id, "both operands must be measures or kernels");
        }
    }
    let source_domain = component_variate(inf, source.0, &source.1);
    let base_domain = component_variate(inf, base.0, &base.1);
    if let (Some(a), Some(b)) = (&source_domain, &base_domain)
        && !compatible(a, b)
    {
        return failed(
            inf,
            id,
            "operands must have the same variate type and shape",
        );
    }
    let mut interface: Option<(NodeId, &[Symbol])> = None;
    for (node, ty, _) in args {
        if let Type::Kernel { inputs, .. } = ty {
            if let Some((first, names)) = interface {
                if names.len() != inputs.len() || inputs.iter().any(|name| !names.contains(name)) {
                    return failed(inf, id, "kernels must have the same named input interface");
                }
                for &name in names {
                    if let (Some(a), Some(b)) =
                        (input_type(inf, first, name), input_type(inf, *node, name))
                        && !compatible(&a, &b)
                    {
                        return failed(inf, id, "kernels have incompatible input types");
                    }
                }
            } else {
                interface = Some((*node, inputs));
            }
        }
    }
    match interface {
        Some((_, inputs)) => Type::Kernel {
            inputs: inputs.into(),
            mass: Mass::Normalized,
        },
        None => Type::Measure {
            domain: Box::new(base_domain.or(source_domain).unwrap_or(Type::Deferred)),
            mass: Mass::Normalized,
        },
    }
}

pub(super) fn result_type(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Option<Type> {
    let mut node = node;
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                node = inf.module.binding(inf.module.binding_by_name(r.name)?).rhs;
            }
            Node::Call(call)
                if matches!(call.head, CallHead::Builtin(op)
                if inf.module.resolve(op) == "tilt") =>
            {
                let base = *call.args.get(1)?;
                let ty = inf.lookup_type(base)?.clone();
                return component_variate(inf, base, &ty).map(|domain| Type::Measure {
                    domain: Box::new(domain),
                    mass: Mass::Normalized,
                });
            }
            _ => return None,
        }
    }
}

fn input_type(inf: &Inferencer<'_, '_>, node: NodeId, name: Symbol) -> Option<Type> {
    let (reification, body) = local_reification(inf, node)?;
    let reference = input_entries(inf, reification)?
        .into_iter()
        .find(|(input, _)| *input == name)?
        .1;
    match reference.ns {
        RefNs::SelfMod => {
            let rhs = inf
                .module
                .binding(inf.module.binding_by_name(reference.name)?)
                .rhs;
            inf.lookup_type(rhs).cloned()
        }
        RefNs::Local => collect_local_refs(inf, body)
            .get(&reference.name)?
            .iter()
            .find_map(|&node| inf.lookup_type(node).cloned()),
        _ => None,
    }
}

fn compatible(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_), _)
        | (_, Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_)) => true,
        (Type::Array { shape: a, elem: x }, Type::Array { shape: b, elem: y }) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(a, b)| a == b || matches!(a, Dim::Dynamic) || matches!(b, Dim::Dynamic))
                && compatible(x, y)
        }
        _ => a == b,
    }
}

fn failed(inf: &mut Inferencer<'_, '_>, id: NodeId, reason: &str) -> Type {
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!("`tilt`: {reason} (spec §06)"),
    ));
    Type::Failed(format!("tilt: {reason}").into())
}
