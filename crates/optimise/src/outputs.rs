//! Map named aggregate outputs to individual roots without flattening the ABI.

use std::collections::HashMap;

use flatppl_core::graph::{rebuild_with_children, resolve_ref};
use flatppl_core::{BindingId, CallHead, DEFAULT_MAX_DEPTH, Module, Node, NodeId, RefNs};

use crate::project::Cuts;

enum Output {
    Slot(NodeId, usize),
    Fields(NodeId, Vec<usize>),
    Named(NodeId, BindingId, usize),
}

#[derive(Default)]
pub(super) struct Outputs {
    pub slots: Vec<NodeId>,
    nodes: Vec<Output>,
    by_source: HashMap<NodeId, usize>,
    bindings: Vec<(BindingId, usize)>,
}

impl Outputs {
    pub fn collect(
        m: &Module,
        cuts: &Cuts,
        follow_names: bool,
        bindings: &[(BindingId, NodeId)],
    ) -> Option<Self> {
        let mut outputs = Self::default();
        for &(binding, id) in bindings {
            let index = outputs.visit(m, cuts, follow_names, id, 0)?;
            outputs.bindings.push((binding, index));
        }
        Some(outputs)
    }

    fn visit(
        &mut self,
        m: &Module,
        cuts: &Cuts,
        follow_names: bool,
        id: NodeId,
        depth: usize,
    ) -> Option<usize> {
        if let Some(&slot) = self.by_source.get(&id) {
            return Some(slot);
        }
        if depth >= DEFAULT_MAX_DEPTH {
            return None;
        }
        let output = if follow_names
            && matches!(m.node(id),Node::Ref(r) if r.ns==RefNs::SelfMod)
            && let Some(rhs) = cuts.resolve(m, id)
            && aggregate(m, cuts, rhs)
            && let (_, Some(binding)) = resolve_ref(m, id)
        {
            let fields = self.visit(m, cuts, follow_names, rhs, depth + 1)?;
            Output::Named(id, binding, fields)
        } else if aggregate(m, cuts, id) {
            let fields = m
                .node(id)
                .children()
                .into_iter()
                .map(|child| self.visit(m, cuts, follow_names, child, depth + 1))
                .collect::<Option<_>>()?;
            Output::Fields(id, fields)
        } else {
            let slot = self.slots.len();
            self.slots.push(id);
            Output::Slot(id, slot)
        };
        let index = self.nodes.len();
        self.nodes.push(output);
        self.by_source.insert(id, index);
        Some(index)
    }

    pub fn write(self, m: &mut Module, chosen: &[Option<NodeId>]) {
        let mut rebuilt = Vec::with_capacity(self.nodes.len());
        for output in self.nodes {
            let node = match output {
                Output::Slot(id, slot) => chosen[slot].unwrap_or(id),
                Output::Named(reference, binding, index) => {
                    m.set_binding_rhs(binding, rebuilt[index]);
                    reference
                }
                Output::Fields(id, fields) => {
                    let children: Vec<_> = fields.into_iter().map(|slot| rebuilt[slot]).collect();
                    if children == m.node(id).children() {
                        id
                    } else {
                        rebuild_with_children(m, id, &children)
                    }
                }
            };
            rebuilt.push(node);
        }
        for (binding, index) in self.bindings {
            m.set_binding_rhs(binding, rebuilt[index]);
        }
    }
}

fn aggregate(m: &Module, cuts: &Cuts, id: NodeId) -> bool {
    !cuts.blocked(m, id)
        && !m.is_sample_result(id)
        && m.sample_origin(id).is_none()
        && matches!(m.node(id),Node::Call(c) if c.inputs.is_none() &&
            matches!(c.head,CallHead::Builtin(h) if matches!(m.resolve(h),"tuple"|"record"|"table")))
}
