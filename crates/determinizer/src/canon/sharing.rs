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

    let closed = lift_scopes.then(|| closed_numeric_candidates(m, &order));
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
        let replacement =
            if uses[id.index()] > 1 && closed.as_ref().is_some_and(|nodes| nodes[id.index()]) {
                changed = true;
                bind(m, rhs, "__shared")
            } else if uses[id.index()] > 1
                && is_numeric_broadcast(m, id)
                && !crate::kernel::has_free_local(m, rhs)
                && !contains_axis(m, rhs)
                && !contains_integral(m, rhs)
            {
                changed = true;
                bind(m, rhs, "__broadcast")
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

/// Name closed numeric expressions only at the serialization boundary. Descendants of a
/// callable or quadrature binder stay in place even if reused outside it.
fn closed_numeric_candidates(m: &Module, order: &[NodeId]) -> Vec<bool> {
    let is_binder = |id| match m.node(id) {
        Node::Call(call) => {
            call.inputs.is_some()
                || matches!(call.head, CallHead::Builtin(h)
                    if matches!(m.resolve(h), flatppl_core::LOG_INTEGRAL | flatppl_core::MASS_INTEGRAL))
        }
        _ => false,
    };
    let mut fenced = vec![false; m.node_count()];
    for &id in order.iter().rev() {
        if fenced[id.index()] || is_binder(id) {
            m.for_each_child(id, |child| fenced[child.index()] = true);
        }
    }
    let mut open = vec![false; m.node_count()];
    let mut candidates = vec![false; m.node_count()];
    for &id in order {
        let mut lexical = is_binder(id)
            || matches!(
                m.node(id),
                Node::Axis(_)
                    | Node::Ref(Ref {
                        ns: RefNs::Local,
                        ..
                    })
            )
            || matches!(m.node(id), Node::Call(call)
                if matches!(call.head, CallHead::Builtin(h)
                    if matches!(m.resolve(h), "elementof" | "external" | "load_data" | flatppl_core::INTEGRATION_POINT)));
        m.for_each_child(id, |child| lexical |= open[child.index()]);
        open[id.index()] = lexical;
        candidates[id.index()] = !lexical
            && !fenced[id.index()]
            && matches!(m.node(id), Node::Call(_))
            && match m.type_of(id) {
                Some(Type::Scalar(_)) => true,
                Some(Type::Array { elem, .. }) => matches!(elem.as_ref(), Type::Scalar(_)),
                _ => false,
            };
    }
    candidates
}

fn is_numeric_broadcast(m: &Module, id: NodeId) -> bool {
    matches!(m.node(id), Node::Call(c)
        if matches!(c.head, CallHead::Builtin(op) if m.resolve(op) == "broadcast"))
        && matches!(m.type_of(id), Some(Type::Array { elem, .. })
            if matches!(elem.as_ref(), Type::Scalar(_)))
}

fn bind(m: &mut Module, rhs: NodeId, prefix: &str) -> NodeId {
    let mut suffix = m.binding_count();
    let name = loop {
        let name = m.intern(&format!("{prefix}_{suffix}"));
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
                if matches!(m.resolve(h), flatppl_core::LOG_INTEGRAL | flatppl_core::MASS_INTEGRAL | flatppl_core::INTEGRATION_POINT))
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

#[cfg(test)]
mod tests {
    use super::super::prepare_serialization;
    use flatppl_core::graph::build_call;

    #[test]
    fn source_serialization_preserves_sharing_in_a_numeric_dag() {
        let mut module =
            flatppl_syntax::parse("x = elementof(reals)\ninputs = x\noutputs = x").unwrap();
        let name = module.intern("outputs");
        let output = module.binding_by_name(name).unwrap();
        let mut value = module.binding(output).rhs;
        for _ in 0..16 {
            value = build_call(&mut module, "add", &[value, value]);
        }
        module.set_binding_rhs(output, value);
        assert!(flatppl_infer::infer(&mut module).is_empty());

        prepare_serialization(&mut module, None);

        let text = flatppl_syntax::print(&module);
        assert!(text.len() < 2_048, "serialized size {}", text.len());
        let mut reparsed = flatppl_syntax::parse(&text).unwrap();
        assert!(flatppl_infer::infer(&mut reparsed).is_empty());
    }
}
