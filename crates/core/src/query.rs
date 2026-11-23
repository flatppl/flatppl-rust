//! The declared input boundary of a FlatPPL query.

use crate::{BindingId, CallHead, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol, Type};
use std::collections::HashSet;

/// A component that the query ABI binds to its own value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Component {
    Field(String),
    Index(usize),
}

/// Read a statically named record/table field or a fixed tuple index. Tensor
/// indexing is an operation on a value, not a path to an ABI component.
pub fn component(m: &Module, id: NodeId) -> Option<(NodeId, Component)> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(head) = c.head else {
        return None;
    };
    let [base, key] = *c.args else {
        return None;
    };
    if !c.named.is_empty() || c.inputs.is_some() {
        return None;
    }
    let head = m.resolve(head);
    let part = match (m.type_of(base)?, m.node(key), head) {
        (
            Type::Record(_) | Type::Table { .. },
            Node::Lit(Scalar::Str(s)),
            "get" | "get0" | "get_field",
        ) => Component::Field(s.to_string()),
        (Type::Tuple(_), Node::Lit(Scalar::Int(i)), "get" | "get0" | "tuple_get") => {
            Component::Index(usize::try_from(i.checked_sub(i64::from(head == "get"))?).ok()?)
        }
        _ => return None,
    };
    Some((base, part))
}

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
