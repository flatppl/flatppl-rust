//! Recover lexical surface callbacks from compiler-only integral coordinates.

use std::collections::{HashMap, HashSet};

use flatppl_core::graph::{build_call, build_record, expect_builtin_call, rebuild_with_children};
use flatppl_core::{
    Call, CallHead, INTEGRATION_POINT, Idx, Inputs, LOG_INTEGRAL, MASS_INTEGRAL, Module, Node,
    NodeId, Phase, Ref, RefNs, Scalar, Symbol,
};

use crate::RefuseError;
use crate::driver::map_tree;

/// Reify numerical integrals as weighted Lebesgue masses for source output.
///
/// Call after preparing serialization, so closed integrals shared by multiple
/// callback bodies remain module bindings. The reified source requires numerical
/// integration to be enabled when lowered again.
pub fn reify_numerical_integrals(module: &mut Module) -> Result<(), RefuseError> {
    let mut names: HashSet<Symbol> = module.bindings().map(|(_, binding)| binding.name).collect();
    for idx in 0..module.node_count() {
        let node = NodeId::from_usize(idx);
        match module.node(node) {
            Node::Ref(reference) => {
                names.insert(reference.name);
            }
            Node::Call(call) => {
                if let Some(Inputs::Spec(entries)) = &call.inputs {
                    names.extend(entries.iter().flat_map(|(label, r)| [*label, r.name]));
                }
                if let Some(entries) = module.auto_inputs_of(node) {
                    names.extend(entries.iter().flat_map(|(label, r)| [*label, r.name]));
                }
            }
            _ => {}
        }
    }
    let roots: Vec<_> = module
        .bindings()
        .map(|(id, binding)| (id, binding.rhs))
        .collect();
    let mut rewritten = HashMap::new();
    for (binding, root) in roots {
        let expression = rewrite(module, root, &mut names, &mut rewritten)?;
        module.set_binding_rhs(binding, expression);
    }
    Ok(())
}

fn rewrite(
    module: &mut Module,
    root: NodeId,
    names: &mut HashSet<Symbol>,
    rewritten: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, RefuseError> {
    if let Some(&value) = rewritten.get(&root) {
        return Ok(value);
    }
    let op = match module.node(root) {
        Node::Call(call) => match call.head {
            CallHead::Builtin(head) => module.resolve(head),
            _ => "",
        },
        _ => "",
    };
    let result = match op {
        INTEGRATION_POINT => {
            return Err(refuse(root, "an integration coordinate escaped its binder"));
        }
        LOG_INTEGRAL | MASS_INTEGRAL => {
            let logarithmic = op == LOG_INTEGRAL;
            let Node::Call(call) = module.node(root) else {
                unreachable!();
            };
            let args = call.args.clone();
            let value = mass_expression(module, &args, logarithmic, names)
                .ok_or_else(|| refuse(root, "expected one or two bound integration coordinates"))?;
            rewrite(module, value, names, rewritten)?
        }
        _ => {
            let children = module.node(root).children();
            let values = children
                .iter()
                .map(|&child| rewrite(module, child, names, rewritten))
                .collect::<Result<Vec<_>, _>>()?;
            if children == values {
                root
            } else {
                rebuild_with_children(module, root, &values)
            }
        }
    };
    rewritten.insert(root, result);
    Ok(result)
}

fn mass_expression(
    module: &mut Module,
    args: &[NodeId],
    logarithmic: bool,
    names: &mut HashSet<Symbol>,
) -> Option<NodeId> {
    let (&body, axes) = args.split_first()?;
    let (axes, extra) = axes.as_chunks::<3>();
    if !extra.is_empty() || !matches!(axes.len(), 1 | 2) {
        return None;
    }
    let (label, reference) = fresh_input(module, "integral_coordinate", names)?;
    let variate = module.alloc(Node::Ref(reference));
    let mut coordinates = Vec::with_capacity(axes.len());
    let mut intervals = Vec::with_capacity(axes.len());
    for (axis, &[point, lower, upper]) in axes.iter().enumerate() {
        expect_builtin_call(module, point, INTEGRATION_POINT)?;
        let value = if axes.len() == 1 {
            variate
        } else {
            let index = module.alloc(Node::Lit(Scalar::Int(axis as i64 + 1)));
            build_call(module, "get", &[variate, index])
        };
        coordinates.push((point, value));
        intervals.push(build_call(module, "interval", &[lower, upper]));
    }
    let mut inputs = vec![(label, reference)];
    let mut substitutions = Vec::new();
    let captures = captures(module, body);
    let parameters = if captures.is_empty() {
        None
    } else {
        let (label, reference) = fresh_input(module, "integral_parameters", names)?;
        inputs.push((label, reference));
        let parameter = module.alloc(Node::Ref(reference));
        let mut fields = Vec::with_capacity(captures.len());
        for (index, (captured, value)) in captures.into_iter().enumerate() {
            let field = format!("capture_{index}");
            let field_name = module.intern(&field);
            fields.push((field_name, value));
            let key = module.alloc(Node::Lit(Scalar::Str(field.into())));
            let local = build_call(module, "get", &[parameter, key]);
            substitutions.push((captured, local));
        }
        Some(build_record(module, &fields))
    };
    let body = map_tree(module, body, &mut |module, id| {
        if let Some(&(_, value)) = coordinates.iter().find(|(point, _)| *point == id) {
            return Some(value);
        }
        match module.node(id) {
            Node::Call(call) if call.inputs.is_some() => Some(id),
            Node::Ref(reference) => substitutions
                .iter()
                .find(|(captured, _)| captured == reference)
                .map(|&(_, local)| local),
            _ => None,
        }
    });
    let function = module.intern("functionof");
    let weight = module.alloc(Node::Call(Call {
        head: CallHead::Builtin(function),
        args: Box::new([body]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    let support = if intervals.len() == 1 {
        intervals[0]
    } else {
        build_call(module, "cartprod", &intervals)
    };
    let reference = build_call(module, "Lebesgue", &[support]);
    let weighting = if logarithmic {
        "logweighted"
    } else {
        "weighted"
    };
    let args: Vec<_> = [weight, reference].into_iter().chain(parameters).collect();
    let measure = build_call(module, weighting, &args);
    let mass = build_call(module, "totalmass", &[measure]);
    Some(if logarithmic {
        build_call(module, "log", &[mass])
    } else {
        mass
    })
}

// An explicit reification boundary has to declare live parameters. Pass each
// captures together as its optional parameter record, including placeholders
// owned by an outer callback. Nested reifications supply their own boundary.
fn captures(module: &Module, root: NodeId) -> Vec<(Ref, NodeId)> {
    let mut captures = Vec::new();
    let mut pending = vec![root];
    let mut visited = HashSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        match module.node(id) {
            Node::Ref(reference) => {
                let capture = match reference.ns {
                    RefNs::Local => true,
                    RefNs::SelfMod => module
                        .binding_by_name(reference.name)
                        .is_some_and(|id| module.binding_phase(id) != Some(Phase::Fixed)),
                    RefNs::Module(_) => module.phase_of(id) == Some(Phase::Parametric),
                };
                if capture && !captures.iter().any(|(existing, _)| existing == reference) {
                    captures.push((*reference, id));
                }
            }
            Node::Call(call) if call.inputs.is_some() => {}
            node => node.for_each_child(|child| pending.push(child)),
        }
    }
    captures
}

fn fresh_input(
    module: &mut Module,
    prefix: &str,
    names: &mut HashSet<Symbol>,
) -> Option<(Symbol, Ref)> {
    (module.node_count()..).find_map(|suffix| {
        let label = module.intern(&format!("{prefix}_{suffix}"));
        let name = module.intern(&format!("_{prefix}_{suffix}_"));
        if names.contains(&label) || names.contains(&name) {
            return None;
        }
        names.extend([label, name]);
        Some((
            label,
            Ref {
                ns: RefNs::Local,
                name,
            },
        ))
    })
}

fn refuse(node: NodeId, reason: &str) -> RefuseError {
    RefuseError {
        node,
        construct: "NumericalIntegral".into(),
        reason: reason.into(),
    }
}
