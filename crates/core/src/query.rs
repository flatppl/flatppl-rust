//! The declared input boundary of a FlatPPL query.

use crate::{BindingId, CallHead, Module, Node, NodeId, Ref, RefNs, Symbol};
use std::collections::HashSet;

/// Return whether this binding declares which values the caller supplies.
pub fn is_inputs_binding(m: &Module, bid: BindingId) -> bool {
    m.resolve(m.binding(bid).name) == "inputs"
}

/// Return every declared query input and every name on its alias chain. A promoted
/// input replaces the original value, including a fixed or derived right-hand side.
pub fn promoted_input_names(m: &Module) -> HashSet<Symbol> {
    let Some((_, binding)) = m.bindings().find(|(_, b)| m.resolve(b.name) == "inputs") else {
        return HashSet::new();
    };
    abi_tuple_elems(m, binding.rhs)
        .into_iter()
        .filter_map(|elem| match m.node(elem) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => Some(*name),
            _ => None,
        })
        .flat_map(|name| m.alias_chain(name))
        .collect()
}

fn abi_tuple_elems(m: &Module, rhs: NodeId) -> Vec<NodeId> {
    if let Node::Call(c) = m.node(rhs)
        && matches!(c.head, CallHead::Builtin(sym) if m.resolve(sym) == "tuple")
    {
        return c.args.to_vec();
    }
    vec![rhs]
}
