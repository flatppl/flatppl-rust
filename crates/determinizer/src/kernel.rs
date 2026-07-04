//! Kernel resolution + application shared by the `kchain` marginal
//! (`marginal.rs`) and the `jointchain` product (`jointchain.rs`).
//!
//! A `kernelof(body, %specinputs([(name, ref), …]))` reifies a measure `body`
//! with named boundary inputs. Each entry is `(name, Ref)`: `name` is what the
//! kernel input is called (matched to a prior variate field by auto-splat);
//! `Ref` is how the body references it — the SAME symbol as `name` for a
//! real-binding input (`(a (%ref self a))`), a placeholder (`(b (%ref %local
//! _b_))`) for an intermediate-variate input. Substitution replaces the `Ref`'s
//! symbol, so callers apply `substitute_ref(body, ref.name, value)`.

use crate::density::resolve_ref_one;
use flatppl_core::{Call, CallHead, Inputs, Module, NamedArg, Node, NodeId, Ref, RefNs, Symbol};

/// A resolved kernel: its reified body and its boundary inputs as
/// `(name, body-target-ref)` pairs, in `%specinputs` order.
pub(crate) struct Kernel {
    pub body: NodeId,
    pub inputs: Vec<(Symbol, Ref)>,
}

/// Resolve `k_arg` to a `kernelof(body, %specinputs([(name, ref), …]))`.
/// `None` for any non-`kernelof` shape or a `kernelof` without a `%specinputs`
/// boundary. Returns ALL inputs; callers that require exactly one check the
/// length themselves.
pub(crate) fn resolve_kernel(m: &Module, k_arg: NodeId) -> Option<Kernel> {
    let (resolved, _) = resolve_ref_one(m, k_arg);
    let Node::Call(c) = m.node(resolved) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "kernelof" || c.args.len() != 1 {
        return None;
    }
    let body = c.args[0];
    let inputs: Vec<(Symbol, Ref)> = match &c.inputs {
        Some(Inputs::Spec(entries)) if !entries.is_empty() => {
            entries.iter().map(|(nm, r)| (*nm, *r)).collect()
        }
        _ => return None,
    };
    Some(Kernel { body, inputs })
}

/// Replace every `(%ref self name)` / `(%ref %local name)` in the subtree at
/// `root` with `new_id`. Append-only. Scope-UNAWARE: sound under the workspace
/// no-shadowing assumption (a substituted symbol is never rebound inside the
/// subtree).
pub(crate) fn substitute_ref(m: &mut Module, root: NodeId, name: Symbol, new_id: NodeId) -> NodeId {
    if let Node::Ref(Ref { ns, name: rname }) = m.node(root) {
        if matches!(ns, RefNs::SelfMod | RefNs::Local) && *rname == name {
            return new_id;
        }
    }
    let children: Vec<NodeId> = m.node(root).children();
    if children.is_empty() {
        return root;
    }
    let new_children: Vec<NodeId> = children
        .iter()
        .map(|&c| substitute_ref(m, c, name, new_id))
        .collect();
    if new_children == children {
        return root;
    }
    let Node::Call(orig) = m.node(root) else {
        unreachable!("non-call node with children is impossible in this IR");
    };
    let head = orig.head;
    let inputs = orig.inputs.clone();
    let n_args = orig.args.len();
    let (new_head, slice) = match head {
        CallHead::User(_) => (CallHead::User(new_children[0]), &new_children[1..]),
        CallHead::Builtin(s) => (CallHead::Builtin(s), &new_children[..]),
    };
    let new_args: Vec<NodeId> = slice[..n_args].to_vec();
    let new_named_values = &slice[n_args..];
    let new_named: Vec<NamedArg> = orig
        .named
        .iter()
        .zip(new_named_values.iter())
        .map(|(na, &val)| NamedArg {
            kind: na.kind,
            name: na.name,
            value: val,
        })
        .collect();
    m.alloc(Node::Call(Call {
        head: new_head,
        args: new_args.into(),
        named: new_named.into(),
        inputs,
    }))
}
