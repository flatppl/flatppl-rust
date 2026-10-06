//! Preserve shared tensor broadcasts when serializing the lowered DAG as a tree.
//! Run after function reduction: naming intermediates earlier can hide parameter
//! occurrences from the syntactic beta reducer.

use std::collections::HashSet;

use flatppl_core::{Binding, CallHead, Idx, Module, Node, NodeId, Ref, RefNs, Type};

mod scoped;

/// Materialize each repeated numeric broadcast once. This deliberately leaves
/// ABI declarations and structural records alone. Cell-local work moves only
/// through a scope-preserving broadcast over the original collection.
pub(super) fn share_broadcasts(m: &mut Module, lift_scopes: bool) -> bool {
    let bindings: Vec<_> = m
        .bindings()
        .filter(|(bid, _)| !super::is_reserved_abi_binding(m, *bid))
        .map(|(bid, b)| (bid, b.rhs))
        .collect();
    let mut uses = vec![0usize; m.node_count()];
    let mut seen = vec![false; m.node_count()];
    let mut order = Vec::new();
    let mut stack = Vec::new();
    for &(_, root) in &bindings {
        uses[root.index()] += 1;
        stack.push((root, false));
    }
    while let Some((id, finish)) = stack.pop() {
        if finish {
            order.push(id);
        } else if !seen[id.index()] {
            seen[id.index()] = true;
            stack.push((id, true));
            m.for_each_child(id, |child| {
                uses[child.index()] += 1;
                stack.push((child, false));
            });
        }
    }

    let mut mapped: Vec<Option<NodeId>> = vec![None; m.node_count()];
    let mut changed = false;
    for id in order {
        let children = m.node(id).children();
        let replacements: Vec<_> = children
            .iter()
            .map(|child| mapped[child.index()].expect("children precede parents"))
            .collect();
        let rhs = if children == replacements {
            id
        } else {
            crate::driver::rebuild_with_children(m, id, &replacements)
        };
        let rhs = if lift_scopes
            && is_numeric_broadcast(m, rhs)
            && !contains_integral(m, rhs)
            && let Some(lifted) = scoped::lift_broadcast(m, rhs)
        {
            changed = true;
            lifted
        } else {
            rhs
        };
        let replacement = if uses[id.index()] > 1
            && is_numeric_broadcast(m, id)
            && !crate::kernel::has_free_local(m, rhs)
            && !contains_axis(m, rhs)
            && !contains_integral(m, rhs)
        {
            changed = true;
            bind(m, rhs)
        } else {
            rhs
        };
        mapped[id.index()] = Some(replacement);
    }
    if changed {
        for (bid, rhs) in bindings {
            m.set_binding_rhs(bid, mapped[rhs.index()].expect("binding root visited"));
        }
    }
    changed
}

fn is_numeric_broadcast(m: &Module, id: NodeId) -> bool {
    matches!(m.node(id), Node::Call(c)
        if matches!(c.head, CallHead::Builtin(op) if m.resolve(op) == "broadcast"))
        && matches!(m.type_of(id), Some(Type::Array { elem, .. })
            if matches!(elem.as_ref(), Type::Scalar(_)))
}

fn bind(m: &mut Module, rhs: NodeId) -> NodeId {
    let mut suffix = m.binding_count();
    let name = loop {
        let name = m.intern(&format!("__broadcast_{suffix}"));
        if m.binding_by_name(name).is_none() {
            break name;
        }
        suffix += 1;
    };
    m.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public: false,
        synthetic: true,
    });
    m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }))
}

// Even a nested aggregate stays in place. Its axes are lexical, not module inputs.
fn contains_axis(m: &Module, root: NodeId) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if matches!(m.node(id), Node::Axis(_)) {
            return true;
        }
        m.for_each_child(id, |child| stack.push(child));
    }
    false
}

fn contains_integral(m: &Module, root: NodeId) -> bool {
    let mut pending = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Node::Call(c) = m.node(id)
            && matches!(c.head, CallHead::Builtin(h)
                if matches!(m.resolve(h), flatppl_core::LOG_INTEGRAL | flatppl_core::INTEGRATION_POINT))
        {
            return true;
        }
        if let Node::Ref(r) = m.node(id)
            && r.ns == RefNs::SelfMod
            && let Some(binding) = m.binding_by_name(r.name)
        {
            pending.push(m.binding(binding).rhs);
        }
        m.for_each_child(id, |child| pending.push(child));
    }
    false
}
