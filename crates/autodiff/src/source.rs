//! Compact shared callable dataflow for source output.

use std::collections::{BTreeSet, HashMap, HashSet};

use flatppl_core::graph::rebuild_with_children;
use flatppl_core::{Binding, Call, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs};

/// Small callbacks are easier to read directly. Above this expanded node
/// count, preserve their DAG through ordinary, explicitly bound FlatPPL calls.
const EXPANDED_BODY_LIMIT: usize = 4096;
const MIN_SHARED_EXPRESSION_SIZE: usize = 600;

pub(super) fn factor_callables(module: &mut Module) {
    let original: Vec<_> = module
        .bindings()
        .map(|(id, binding)| (id, binding.clone()))
        .collect();
    let existing: HashSet<_> = original.iter().map(|(id, _)| *id).collect();
    let mut rewritten = HashMap::new();
    for (id, binding) in &original {
        let root = rewrite(module, binding.rhs, &mut rewritten);
        module.set_binding_rhs(*id, root);
    }
    let helpers: HashSet<_> = module
        .bindings()
        .filter_map(|(id, _)| (!existing.contains(&id)).then_some(id))
        .collect();
    if helpers.is_empty() {
        return;
    }
    // The inliner reads bindings in source order. Definitions are built
    // tail-first, so every helper must precede the original call site.
    let originals: Vec<_> = original
        .into_iter()
        .map(|(id, _)| module.binding(id).clone())
        .collect();
    module.retain_bindings(&helpers);
    for binding in originals {
        module.add_binding(binding);
    }
}

fn rewrite(module: &mut Module, node: NodeId, memo: &mut HashMap<NodeId, NodeId>) -> NodeId {
    if let Some(&result) = memo.get(&node) {
        return result;
    }
    let children = module.node(node).children();
    let values: Vec<_> = children
        .iter()
        .map(|&child| rewrite(module, child, memo))
        .collect();
    let result = if let Node::Call(call) = module.node(node)
        && matches!(call.head, CallHead::Builtin(name) if module.resolve(name) == "functionof")
        && let Some(Inputs::Spec(formals)) = &call.inputs
        && formals.iter().all(|(_, formal)| formal.ns == RefNs::Local)
        && let [body] = *values.as_slice()
        && let Some(body) = factor_body(module, body)
    {
        rebuild_with_children(module, node, &[body])
    } else if children == values {
        node
    } else {
        rebuild_with_children(module, node, &values)
    };
    memo.insert(node, result);
    result
}

fn factor_body(module: &mut Module, body: NodeId) -> Option<NodeId> {
    let mut uses = HashMap::new();
    let mut costs = HashMap::new();
    let mut order = Vec::new();
    uses.insert(body, 1usize);
    let (size, _) = visit(module, body, &mut uses, &mut costs, &mut order)?;
    if size <= EXPANDED_BODY_LIMIT {
        return None;
    }
    let mut terms = HashMap::new();
    let mut definitions: Vec<HashMap<flatppl_core::Symbol, NodeId>> = Vec::new();
    let mut depths = HashMap::<NodeId, usize>::new();
    let mut printed: HashMap<NodeId, usize> = HashMap::new();
    for node in order {
        let children = module.node(node).children();
        let values: Vec<_> = children.iter().map(|child| terms[child]).collect();
        let expression = if children == values {
            node
        } else {
            rebuild_with_children(module, node, &values)
        };
        let (_, free_axis) = costs[&node];
        let size = children
            .iter()
            .fold(1usize, |size, child| size.saturating_add(printed[child]));
        let depth = children
            .iter()
            .map(|child| depths[child])
            .max()
            .unwrap_or(0);
        if uses[&node] > 1 && size >= MIN_SHARED_EXPRESSION_SIZE && !free_axis {
            let name = module.intern(&format!("_ad_shared_value_{}_", module.node_count()));
            let reference = Ref {
                ns: RefNs::Local,
                name,
            };
            let variable = module.alloc(Node::Ref(reference));
            terms.insert(node, variable);
            printed.insert(node, 1);
            if depth >= definitions.len() {
                definitions.resize_with(depth + 1, HashMap::new);
            }
            definitions[depth].insert(reference.name, expression);
            depths.insert(node, depth + 1);
        } else {
            terms.insert(node, expression);
            printed.insert(node, size);
            depths.insert(node, depth);
        }
    }
    if definitions.is_empty() {
        return None;
    }
    let mut result = terms[&body];
    for values in definitions.into_iter().rev() {
        let mut captures = BTreeSet::new();
        free_locals(module, result, &mut captures, &mut HashSet::new());
        if captures.iter().any(|name| values.contains_key(name)) {
            result = bind(module, result, &values, captures);
        }
    }
    Some(result)
}

fn visit(
    module: &Module,
    node: NodeId,
    uses: &mut HashMap<NodeId, usize>,
    costs: &mut HashMap<NodeId, (usize, bool)>,
    order: &mut Vec<NodeId>,
) -> Option<(usize, bool)> {
    if let Some(&info) = costs.get(&node) {
        return Some(info);
    }
    let mut free_axis = matches!(module.node(node), Node::Axis(_));
    let mut size = 1usize;
    let Node::Call(call) = module.node(node) else {
        costs.insert(node, (size, free_axis));
        order.push(node);
        return Some((size, free_axis));
    };
    if call.inputs.is_some() {
        return None;
    }
    for child in module.node(node).children() {
        *uses.entry(child).or_default() += 1;
        let (child_size, child_axis) = visit(module, child, uses, costs, order)?;
        size = size.saturating_add(child_size);
        free_axis |= child_axis;
    }
    // The aggregate owns the frame of every axis mentioned in its body.
    if matches!(call.head, CallHead::Builtin(op) if matches!(module.resolve(op), "aggregate" | "metricsum"))
    {
        free_axis = false;
    }
    costs.insert(node, (size, free_axis));
    order.push(node);
    Some((size, free_axis))
}

fn free_locals(
    module: &Module,
    node: NodeId,
    locals: &mut BTreeSet<flatppl_core::Symbol>,
    seen: &mut HashSet<NodeId>,
) {
    if !seen.insert(node) {
        return;
    }
    if let Node::Ref(Ref {
        ns: RefNs::Local,
        name,
    }) = module.node(node)
    {
        locals.insert(*name);
    }
    for child in module.node(node).children() {
        free_locals(module, child, locals, seen);
    }
}

fn bind(
    module: &mut Module,
    body: NodeId,
    values: &HashMap<flatppl_core::Symbol, NodeId>,
    captures: BTreeSet<flatppl_core::Symbol>,
) -> NodeId {
    let formals: Vec<_> = captures
        .iter()
        .enumerate()
        .map(|(index, &name)| {
            (
                module.intern(&format!("arg{index}")),
                Ref {
                    ns: RefNs::Local,
                    name,
                },
            )
        })
        .collect();
    let function = module.intern("functionof");
    let body = module.alloc(Node::Call(Call {
        head: CallHead::Builtin(function),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(formals.into())),
    }));
    let mut suffix = module.node_count();
    let name = loop {
        let name = module.intern(&format!("_ad_callback_{suffix}"));
        if module.binding_by_name(name).is_none() {
            break name;
        }
        suffix += 1;
    };
    module.add_binding(Binding {
        name,
        rhs: body,
        doc: None,
        public: false,
        synthetic: false,
    });
    let function = module.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }));
    let actuals: Vec<_> = captures
        .into_iter()
        .map(|name| {
            if let Some(&expression) = values.get(&name) {
                expression
            } else {
                module.alloc(Node::Ref(Ref {
                    ns: RefNs::Local,
                    name,
                }))
            }
        })
        .collect();
    module.alloc(Node::Call(Call {
        head: CallHead::User(function),
        args: actuals.into(),
        named: Box::new([]),
        inputs: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatppl_core::graph::{build_call, expect_builtin_call};
    use flatppl_infer::Severity;

    #[test]
    fn shared_callback_expressions_remain_compact_and_reparse() {
        let mut module = flatppl_syntax::parse(
            "f(x, y) = x + y\na = elementof(reals)\nb = elementof(reals)\ninputs = (a, b)\noutputs = f(a, b)",
        )
        .unwrap();
        let name = module.intern("f");
        let binding = module.binding_by_name(name).unwrap();
        let function = module.binding(binding).rhs;
        let body = expect_builtin_call(&module, function, "functionof")
            .unwrap()
            .args[0];
        let [mut left, mut right] = *expect_builtin_call(&module, body, "add").unwrap().args else {
            unreachable!()
        };
        for _ in 0..16 {
            left = build_call(&mut module, "add", &[left, left]);
            right = build_call(&mut module, "add", &[right, right]);
        }
        let body = build_call(&mut module, "add", &[left, right]);
        let function = rebuild_with_children(&mut module, function, &[body]);
        module.set_binding_rhs(binding, function);

        factor_callables(&mut module);
        let source = flatppl_syntax::print(&module);
        assert!(source.len() < 16_384);
        assert_eq!(source.matches("_ad_callback_").count(), 2);
        let mut parsed = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut parsed);
        assert!(
            diagnostics
                .iter()
                .all(|diag| diag.severity != Severity::Error),
            "{diagnostics:?}"
        );
    }
}
