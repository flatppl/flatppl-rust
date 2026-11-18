//! Resolve exact log masses after specialization has bound each query's inputs.

use flatppl_core::calling::reified_inputs;
use flatppl_core::graph::{
    build_call, expect_builtin_call, rebuild_with_children, resolve_ref_one,
};
use flatppl_core::{CallHead, MASS_FROM_LOG, Module, Node, NodeId, RefNs, Symbol};
use std::collections::{HashMap, HashSet};

pub(super) fn lower(m: &mut Module) -> bool {
    if crate::driver::find_node(m, |id| expect_builtin_call(m, id, MASS_FROM_LOG).is_some())
        .is_none()
    {
        return false;
    }
    let mut pass = MassReads {
        promoted: crate::canon::promoted_input_names(m),
        boundaries: HashSet::new(),
        mapped: HashMap::new(),
    };
    let roots: Vec<_> = m.bindings().map(|(id, b)| (id, b.rhs)).collect();
    // Every alias must resolve in the original graph until its uses are classified.
    let roots: Vec<_> = roots
        .into_iter()
        .map(|(id, rhs)| (id, pass.rewrite(m, rhs)))
        .collect();
    for (id, rhs) in roots {
        if m.binding(id).rhs != rhs {
            m.set_binding_rhs(id, rhs);
        }
    }
    true
}

struct MassReads {
    promoted: HashSet<Symbol>,
    boundaries: HashSet<NodeId>,
    mapped: HashMap<NodeId, NodeId>,
}

impl MassReads {
    fn log_value(&self, m: &Module, mut value: NodeId) -> Option<NodeId> {
        for _ in 0..=m.binding_count() {
            if self.boundaries.contains(&value) {
                return None;
            }
            if let Some(c) = expect_builtin_call(m, value, MASS_FROM_LOG) {
                return Some(c.args[0]);
            }
            let (next, binding) = resolve_ref_one(m, value);
            if self.promoted.contains(&m.binding(binding?).name) {
                return None;
            }
            value = next;
        }
        None
    }

    fn rewrite(&mut self, m: &mut Module, id: NodeId) -> NodeId {
        if let Some(&value) = self.mapped.get(&id) {
            return value;
        }
        let result = if let Some(c) = expect_builtin_call(m, id, "log")
            && let [arg] = *c.args
            && let Some(value) = self.log_value(m, arg)
        {
            self.rewrite(m, value)
        } else if let Some(c) = expect_builtin_call(m, id, MASS_FROM_LOG) {
            let value = self.rewrite(m, c.args[0]);
            build_call(m, "exp", &[value])
        } else {
            let original = m.node(id).children();
            let mut children = original.clone();
            // A callable may name a body defined outside it, e.g. f=functionof(ell,y=Z)
            // with ell=log(Z). Preserve its supplied Z while the outer ell uses log evidence.
            if let Node::Call(c) = m.node(id)
                && matches!(c.head, CallHead::Builtin(_))
                && c.args.len() == 1
                && let Some(inputs) = &c.inputs
                && let Some(entries) = reified_inputs(m, id, inputs)
            {
                let targets: Vec<_> = entries
                    .iter()
                    .map(|(_, r)| *r)
                    .filter(|r| {
                        r.ns == RefNs::SelfMod
                            && !self.promoted.contains(&r.name)
                            && m.binding_by_name(r.name)
                                .is_some_and(|bid| self.log_value(m, m.binding(bid).rhs).is_some())
                    })
                    .collect();
                if !targets.is_empty() {
                    let bound: Vec<_> = targets
                        .into_iter()
                        .map(|r| {
                            let value = m.alloc(Node::Ref(r));
                            self.boundaries.insert(value);
                            (r, value)
                        })
                        .collect();
                    children[0] = crate::kernel::substitute_admitted(
                        m,
                        children[0],
                        &bound,
                        crate::kernel::Substitute::All,
                        false,
                    );
                }
            }
            for child in &mut children {
                *child = self.rewrite(m, *child);
            }
            if children == original {
                id
            } else {
                rebuild_with_children(m, id, &children)
            }
        };
        self.mapped.insert(id, result);
        result
    }
}
