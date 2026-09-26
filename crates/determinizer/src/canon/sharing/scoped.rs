//! Distribute a map over shared cell computations without cloning its body.
//!
//! For `f.(xs)`, a repeated array `g(x)` becomes a named `g.(xs)` argument
//! of the final map. Each argument retains the original batch shape, including
//! empty axes. This handles one collection input. General multi-input maps
//! need a shape join before a computation can move outside their cell scope.

use std::collections::{HashMap, HashSet};

use flatppl_core::{Call, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs, Symbol, Type};

use crate::{density::resolve_ref_one, driver::rebuild_with_children, kernel};

struct Input {
    label: Symbol,
    reference: Ref,
    values: NodeId,
}

pub(super) fn lift_broadcast(m: &mut Module, id: NodeId) -> Option<NodeId> {
    let Node::Call(call) = m.node(id) else {
        return None;
    };
    if !matches!(call.head, CallHead::Builtin(op) if m.resolve(op) == "broadcast")
        || call.args.len() != 2
        || !call.named.is_empty()
    {
        return None;
    }
    let function = kernel::resolve_reified(m, call.args[0])?;
    if function.auto || function.inputs.len() != 1 {
        return None;
    }
    let values = call.args[1];
    let (resolved, _) = resolve_ref_one(m, values);
    if !matches!(
        m.type_of(values).or_else(|| m.type_of(resolved)),
        Some(Type::Array { .. })
    ) || kernel::has_free_local(m, values)
        || super::contains_axis(m, values)
    {
        return None;
    }
    let (label, reference) = function.inputs[0];
    let mut inputs = vec![Input {
        label,
        reference,
        values,
    }];
    let (order, uses) = cell_graph(m, function.body);
    let mut mapped = HashMap::new();
    let mut changed = false;
    for id in order {
        // A nested reification owns its scope. Its actual arguments remain
        // visible in this cell, but its body must not acquire outer locals.
        let children = cell_children(m, id);
        let replacements: Vec<_> = children.iter().map(|c| mapped[c]).collect();
        let rhs = if children == replacements {
            id
        } else {
            rebuild_with_children(m, id, &replacements)
        };
        let replacement =
            if uses[&id] > 1 && super::is_numeric_broadcast(m, id) && !super::contains_axis(m, rhs)
            {
                if let Some(batch) = map_cell(m, rhs, &inputs) {
                    let values = super::bind(m, batch);
                    let (label, reference, local) = placeholder(m, m.type_of(id)?.clone(), &inputs);
                    inputs.push(Input {
                        label,
                        reference,
                        values,
                    });
                    changed = true;
                    local
                } else {
                    rhs
                }
            } else {
                rhs
            };
        mapped.insert(id, replacement);
    }
    changed
        .then(|| map_cell(m, mapped[&function.body], &inputs))
        .flatten()
}

fn cell_children(m: &Module, id: NodeId) -> Vec<NodeId> {
    match m.node(id) {
        Node::Call(c) if c.inputs.is_some() => Vec::new(),
        node => node.children(),
    }
}

fn cell_graph(m: &Module, root: NodeId) -> (Vec<NodeId>, HashMap<NodeId, usize>) {
    let mut order = Vec::new();
    let mut uses = HashMap::from([(root, 1)]);
    let mut seen = HashSet::new();
    let mut stack = vec![(root, false)];
    while let Some((id, finish)) = stack.pop() {
        if finish {
            order.push(id);
        } else if seen.insert(id) {
            stack.push((id, true));
            for child in cell_children(m, id) {
                *uses.entry(child).or_default() += 1;
                stack.push((child, false));
            }
        }
    }
    (order, uses)
}

/// All selected arguments are arrays with the same outer shape as `xs`.
/// Omitting an unused input therefore cannot change the broadcast domain.
fn map_cell(m: &mut Module, body: NodeId, inputs: &[Input]) -> Option<NodeId> {
    let (order, _) = cell_graph(m, body);
    let refs: Vec<_> = order
        .iter()
        .filter_map(|id| match m.node(*id) {
            Node::Ref(r) => Some(*r),
            _ => None,
        })
        .collect();
    let used: Vec<_> = inputs
        .iter()
        .filter(|input| refs.contains(&input.reference))
        .collect();
    if used.is_empty() {
        return None;
    }
    let head = CallHead::Builtin(m.intern("functionof"));
    let function = m.alloc(Node::Call(Call {
        head,
        args: vec![body].into(),
        named: vec![].into(),
        inputs: Some(Inputs::Spec(
            used.iter().map(|i| (i.label, i.reference)).collect(),
        )),
    }));
    if kernel::has_free_local(m, function) {
        return None;
    }
    let head = CallHead::Builtin(m.intern("broadcast"));
    let args = std::iter::once(function)
        .chain(used.iter().map(|i| i.values))
        .collect();
    Some(m.alloc(Node::Call(Call {
        head,
        args,
        named: vec![].into(),
        inputs: None,
    })))
}

fn placeholder(m: &mut Module, ty: Type, inputs: &[Input]) -> (Symbol, Ref, NodeId) {
    use flatppl_core::Idx;
    let used: HashSet<_> = (0..m.node_count())
        .filter_map(|i| match m.node(NodeId::from_usize(i)) {
            Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }) => Some(*name),
            _ => None,
        })
        .collect();
    let mut serial = m.node_count();
    let (label, name) = loop {
        let name = m.intern(&format!("_shared_broadcast_{serial}_"));
        let label = m.intern(&format!("shared_broadcast_{serial}"));
        if !used.contains(&name) && inputs.iter().all(|i| i.label != label) {
            break (label, name);
        }
        serial += 1;
    };
    let reference = Ref {
        ns: RefNs::Local,
        name,
    };
    let local = m.alloc(Node::Ref(reference));
    m.set_type(local, ty);
    (label, reference, local)
}
