//! Name closed scalar integrals before entering broadcast or quadrature scopes.

use std::collections::{HashMap, HashSet};

use flatppl_core::{Binding, CallHead, Idx, Inputs, Module, Node, NodeId, Ref, RefNs};

pub(super) fn hoist_integrals(m: &mut Module) -> bool {
    let bindings: Vec<_> = m
        .bindings()
        .filter(|(bid, _)| !super::is_reserved_abi_binding(m, *bid))
        .map(|(bid, b)| (bid, b.rhs))
        .collect();
    let scoped = scoped_integrals(m, bindings.iter().map(|(_, rhs)| *rhs));
    let mut names: HashMap<_, _> = bindings
        .iter()
        .filter(|(_, rhs)| is_integral(m, *rhs))
        .map(|(bid, rhs)| (*rhs, m.binding(*bid).name))
        .collect();
    let mut mapped: Vec<Option<(NodeId, NodeId)>> = vec![None; m.node_count()];
    let mut changed = false;
    for &(bid, root) in &bindings {
        let mut pending = vec![(root, false)];
        while let Some((id, ready)) = pending.pop() {
            if mapped[id.index()].is_some() {
                continue;
            }
            let children = m.node(id).children();
            if !ready {
                pending.push((id, true));
                pending.extend(children.into_iter().map(|child| (child, false)));
                continue;
            }
            let replacements: Vec<_> = children
                .iter()
                .map(|child| mapped[child.index()].expect("children precede parents").1)
                .collect();
            let rhs = if children == replacements {
                id
            } else {
                crate::driver::rebuild_with_children(m, id, &replacements)
            };
            let value = if is_integral(m, rhs) && !scoped.contains(&id) && is_closed(m, rhs, &[]) {
                let name = *names.entry(id).or_insert_with(|| {
                    let mut suffix = m.binding_count();
                    let name = loop {
                        let name = m.intern(&format!("__integral_{suffix}"));
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
                    changed = true;
                    name
                });
                m.alloc(Node::Ref(Ref {
                    ns: RefNs::SelfMod,
                    name,
                }))
            } else {
                rhs
            };
            mapped[id.index()] = Some((rhs, value));
        }
        let (rhs, value) = mapped[root.index()].expect("binding root visited");
        let value = if names.get(&root) == Some(&m.binding(bid).name) {
            rhs
        } else {
            value
        };
        changed |= value != root;
        m.set_binding_rhs(bid, value);
    }
    changed
}

fn is_integral(m: &Module, id: NodeId) -> bool {
    matches!(m.node(id), Node::Call(c)
        if matches!(c.head, CallHead::Builtin(h) if m.resolve(h) == flatppl_core::LOG_INTEGRAL))
}

// A surviving broadcast/scan callable may bind a same-module reference, not
// just a placeholder. A shared integral must be closed at every use site.
fn scoped_integrals(m: &Module, roots: impl Iterator<Item = NodeId>) -> HashSet<NodeId> {
    let mut pending: Vec<_> = roots.map(|id| (id, Vec::<Ref>::new())).collect();
    let mut seen: HashMap<NodeId, Vec<Vec<Ref>>> = HashMap::new();
    let mut scoped = HashSet::new();
    while let Some((id, mut inputs)) = pending.pop() {
        let scopes = seen.entry(id).or_default();
        if scopes.contains(&inputs) {
            continue;
        }
        scopes.push(inputs.clone());
        if is_integral(m, id) && !inputs.is_empty() && !is_closed(m, id, &inputs) {
            scoped.insert(id);
        }
        if let Node::Ref(r) = m.node(id) {
            if r.ns == RefNs::SelfMod
                && !inputs.contains(r)
                && let Some(binding) = m.binding_by_name(r.name)
            {
                pending.push((m.binding(binding).rhs, inputs));
            }
            continue;
        }
        if let Node::Call(c) = m.node(id) {
            let entries = match &c.inputs {
                Some(Inputs::Spec(entries)) => entries.as_ref(),
                Some(Inputs::Auto) => m.auto_inputs_of(id).unwrap_or_default(),
                None => &[],
            };
            for (_, reference) in entries {
                if !inputs.contains(reference) {
                    inputs.push(*reference);
                }
            }
        }
        m.for_each_child(id, |child| pending.push((child, inputs.clone())));
    }
    scoped
}

// Only the integrand binds its coordinate. Module references have module scope.
// A free local or aggregate axis conservatively keeps the integral in place.
fn is_closed(m: &Module, root: NodeId, inputs: &[Ref]) -> bool {
    let mut pending = vec![(root, None)];
    let mut seen = HashSet::new();
    while let Some((id, bound)) = pending.pop() {
        if !seen.insert((id, bound)) {
            continue;
        }
        match m.node(id) {
            Node::Ref(r) if inputs.contains(r) => return false,
            Node::Axis(_)
            | Node::Ref(Ref {
                ns: RefNs::Local, ..
            }) => return false,
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => {
                if let Some(binding) = m.binding_by_name(*name) {
                    pending.push((m.binding(binding).rhs, None));
                }
                continue;
            }
            Node::Call(c)
                if matches!(c.head, CallHead::Builtin(h)
                if m.resolve(h) == flatppl_core::INTEGRATION_POINT) =>
            {
                if bound != Some(id) {
                    return false;
                }
                continue;
            }
            Node::Call(c) if is_integral(m, id) && c.args.len() == 4 => {
                pending.push((c.args[0], Some(c.args[1])));
                pending.extend(c.args[2..].iter().map(|&node| (node, bound)));
                continue;
            }
            _ => (),
        }
        m.for_each_child(id, |child| pending.push((child, bound)));
    }
    true
}
