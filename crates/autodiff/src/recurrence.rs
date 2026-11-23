//! Expose small, fixed scalar recurrences to the existing AD graph.

use std::collections::{HashMap, HashSet};

use flatppl_core::calling::reified_inputs;
use flatppl_core::graph::{
    build_call, expect_builtin_call, rebuild_with_children, resolve_ref_one,
};
use flatppl_core::{Binding, BindingId, DEFAULT_MAX_DEPTH, Dim, Module, Node, NodeId, Ref, RefNs};

use crate::{REAL, tensor};

const MAX_STEPS: u32 = 64;

pub(super) fn lower(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    let (function, initial, sequence, params) =
        if let Some(call) = expect_builtin_call(module, node, "reduce") {
            let [function, sequence, params @ ..] = call.args.as_ref() else {
                return None;
            };
            (*function, None, *sequence, params)
        } else {
            let call = expect_builtin_call(module, node, "scan")?;
            let [function, initial, sequence, params @ ..] = call.args.as_ref() else {
                return None;
            };
            (*function, Some(*initial), *sequence, params)
        };
    let params = params.to_vec();
    let [Dim::Static(length)] = *tensor::real_shape(module, sequence)? else {
        return None;
    };
    if length == 0 || length > MAX_STEPS || params.len() > 1 {
        return None;
    }
    let expected = module.type_of(node)?.clone();
    if initial.map_or(expected != REAL, |value| {
        module.type_of(value) != Some(&REAL)
    }) {
        return None;
    }
    let (body, formals) = callback(module, inputs, function)?;
    if formals.len() != 2 + params.len() {
        return None;
    }

    let start = if initial.is_some() { 1 } else { 2 };
    let mut state = initial.unwrap_or_else(|| element(module, sequence, 1));
    let mut states = Vec::new();
    for index in start..=length {
        let cell = element(module, sequence, index);
        let actual = [state, cell].into_iter().chain(params.iter().copied());
        let bindings: Vec<_> = formals.iter().copied().zip(actual).collect();
        state = substitute(module, inputs, body, &bindings, &mut HashMap::new(), 0)?;
        states.push(state);
    }
    let result = if initial.is_some() {
        build_call(module, "vector", &states)
    } else {
        state
    };
    let (types, diagnostics) = flatppl_infer::infer_expression(module, result, &[]);
    if types.get(&result) != Some(&expected)
        || diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return None;
    }
    for (node, ty) in types {
        if module.type_of(node).is_none() {
            module.set_type(node, ty);
        }
    }

    // A step can read its state multiple times. Bind each new state to keep
    // the source representation linear in the number of iterations.
    let mut named = HashMap::new();
    for state in states {
        if named.contains_key(&state) {
            continue;
        }
        let rhs = replace_states(module, state, &mut named, 0)?;
        let alias = bind_value(module, rhs, "_ad_state")?;
        named.insert(state, alias);
    }
    replace_states(module, result, &mut named, 0)
}

fn callback(
    module: &Module,
    inputs: &HashSet<BindingId>,
    mut function: NodeId,
) -> Option<(NodeId, Vec<Ref>)> {
    for _ in 0..=DEFAULT_MAX_DEPTH {
        let (rhs, Some(binding)) = resolve_ref_one(module, function) else {
            let callable = expect_builtin_call(module, function, "functionof")?;
            let [body] = *callable.args.as_ref() else {
                return None;
            };
            let entries = reified_inputs(module, function, callable.inputs.as_ref()?)?;
            let params = entries.iter().map(|&(_, reference)| reference).collect();
            return Some((body, params));
        };
        if inputs.contains(&binding) {
            return None;
        }
        function = rhs;
    }
    None
}

fn element(module: &mut Module, vector: NodeId, index: u32) -> NodeId {
    let index = tensor::int(module, i64::from(index));
    build_call(module, "get", &[vector, index])
}

fn substitute(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
    bindings: &[(Ref, NodeId)],
    memo: &mut HashMap<NodeId, NodeId>,
    depth: usize,
) -> Option<NodeId> {
    if depth > DEFAULT_MAX_DEPTH {
        return None;
    }
    if let Some(&previous) = memo.get(&node) {
        return Some(previous);
    }
    let replacement = match module.node(node) {
        Node::Ref(reference) => {
            if let Some((_, actual)) = bindings.iter().find(|(formal, _)| formal == reference) {
                *actual
            } else if reference.ns == RefNs::Local {
                return None;
            } else {
                let (rhs, binding) = resolve_ref_one(module, node);
                if binding.is_some_and(|binding| !inputs.contains(&binding)) {
                    let replaced = substitute(module, inputs, rhs, bindings, memo, depth + 1)?;
                    if replaced == rhs { node } else { replaced }
                } else {
                    node
                }
            }
        }
        Node::Call(call) if call.inputs.is_some() => return None,
        _ => {
            let children = module.node(node).children();
            let args = children
                .iter()
                .map(|&child| substitute(module, inputs, child, bindings, memo, depth + 1))
                .collect::<Option<_>>()?;
            rebuild(module, node, children, args)
        }
    };
    memo.insert(node, replacement);
    Some(replacement)
}

fn replace_states(
    module: &mut Module,
    node: NodeId,
    named: &mut HashMap<NodeId, NodeId>,
    depth: usize,
) -> Option<NodeId> {
    if let Some(&reference) = named.get(&node) {
        return Some(reference);
    }
    if depth > DEFAULT_MAX_DEPTH {
        return None;
    }
    let children = module.node(node).children();
    let args = children
        .iter()
        .map(|&child| replace_states(module, child, named, depth + 1))
        .collect::<Option<_>>()?;
    let result = rebuild(module, node, children, args);
    if result != node {
        let ty = module.type_of(node)?.clone();
        module.set_type(result, ty);
    }
    // Do not insert unchanged nodes, since a future state can equal this node.
    if result != node {
        named.insert(node, result);
    }
    Some(result)
}

pub(super) fn bind_value(module: &mut Module, rhs: NodeId, prefix: &str) -> Option<NodeId> {
    let mut index = module.node_count();
    let name = loop {
        let name = module.intern(&format!("{prefix}_{index}"));
        if module.binding_by_name(name).is_none() {
            break name;
        }
        index += 1;
    };
    let ty = module.type_of(rhs)?.clone();
    module.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public: false,
        synthetic: false,
    });
    let reference = module.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }));
    module.set_type(reference, ty);
    Some(reference)
}

fn rebuild(module: &mut Module, node: NodeId, before: Vec<NodeId>, after: Vec<NodeId>) -> NodeId {
    if before == after {
        node
    } else {
        rebuild_with_children(module, node, &after)
    }
}
