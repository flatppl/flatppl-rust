//! Thin IR-construction helpers for the determiniser, mirroring the subset of
//! `crates/hs3/src/builder.rs` that rewriting passes need.

use flatppl_core::{Call, CallHead, Module, NamedArg, Node, NodeId, Ref, RefNs, Scalar};

/// Allocate a positional builtin call `head(args…)`.
#[allow(dead_code)]
pub(crate) fn call(m: &mut Module, head: &str, args: &[NodeId]) -> NodeId {
    let sym = m.intern(head);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: args.to_vec().into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }))
}

/// Allocate a `(%ref self <name>)` node — a reference to a current-module binding.
#[allow(dead_code)]
pub(crate) fn self_ref(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name: sym,
    }))
}

/// Allocate a real-literal node.
#[allow(dead_code)]
pub(crate) fn lit_real(m: &mut Module, v: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(v)))
}
