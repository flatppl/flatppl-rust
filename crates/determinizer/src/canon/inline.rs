//! Pass 2: inline residual user-defined function/kernel calls so no
//! `(%call User ...)` reaches consumers (Buffy #263; unblocks #261).
//!
//! A live `(%call (%ref self scale) 1.5)` is exactly what flatppl-js cannot
//! evaluate on the score path
//! (Buffy #261): FlatPDL is defined as deterministic ops + the six
//! `builtin_*` primitives, and a call to a user function is neither. This
//! pass beta-reduces every such call away, because eliminating one is strictly
//! more useful than merely detecting it.
//!
//! `is_flatpdl` (`conformance.rs`) also rejects a residual `CallHead::User`
//! (`NonConformKind::ResidualUserCall`), so a call this pass CANNOT reduce
//! refuses at the driver's exit gate instead of reporting FlatPDL-conformant.
//! Two shapes reach it: an unresolved callee, and an arity mismatch at the call
//! site.

use flatppl_core::{CallHead, Module, Node, NodeId};

use crate::driver::rebuild_with_children;
use crate::kernel::reduce_kernel_application;

/// Replace each `(%call User(callee) args)` with its beta-reduced body. Reuses
/// `reduce_kernel_application` (which resolves the reified callee, binds inputs
/// by position/keyword/record-splat, and substitutes via `substitute_ref`). A
/// call it cannot reduce (unresolved callee, arity mismatch) is left in place —
/// this pass itself stays refuse-free — and `is_flatpdl` then rejects it at the
/// driver's exit gate, so the module refuses rather than reaching a consuming
/// engine that cannot evaluate it.
pub(crate) fn inline_user_calls(m: &mut Module) -> bool {
    let mut changed = false;
    let pairs: Vec<(flatppl_core::BindingId, NodeId)> =
        m.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    for (bid, root) in pairs {
        // `reduce_kernel_application` needs `&mut Module` and returns a fresh
        // body `NodeId`, so apply it via a manual bottom-up walk rather than
        // `map_tree` (whose closure only gets `&Module`, no `alloc`).
        let new = inline_walk(m, root);
        if new != root {
            m.set_binding_rhs(bid, new);
            changed = true;
        }
    }
    changed
}

/// Bottom-up: inline within children first (so a callee itself containing a
/// user call is reduced before we look at this node), then attempt to reduce
/// this node if it is a user call. Rebuilds via `rebuild_with_children` for
/// the child layer — the same rebuild `map_tree` uses, kept in one place so
/// both stay consistent in how a `Call`'s children decode back into
/// head/args/named.
fn inline_walk(m: &mut Module, id: NodeId) -> NodeId {
    let children: Vec<NodeId> = m.node(id).children();
    let mut any_child_changed = false;
    let new_children: Vec<NodeId> = children
        .iter()
        .map(|&c| {
            let nc = inline_walk(m, c);
            any_child_changed |= nc != c;
            nc
        })
        .collect();
    let id = if any_child_changed {
        rebuild_with_children(m, id, &new_children)
    } else {
        id
    };
    // Then reduce this node if it is a user call.
    if let Node::Call(c) = m.node(id) {
        if matches!(c.head, CallHead::User(_)) {
            if let Some(head) = builtin_callee_head(m, id) {
                return rebuild_with_head(m, id, head);
            }
            if let Some(reduced) = reduce_kernel_application(m, id) {
                // The reduced body may itself contain further user calls
                // (e.g. a function whose body calls another function).
                return inline_walk(m, reduced);
            }
        }
    }
    id
}

/// If `id` is `(%call <bare builtin> args…)`, the `Builtin` head that same application
/// spells directly. A [`Node::Const`] is a bare built-in symbol (user bindings are `Ref`),
/// so applying one IS a builtin call: `(%call log 0.5)` and `(log 0.5)` denote the same
/// thing, but only the latter is FlatPDL and only the latter carries a resolved type —
/// `reduce_kernel_application` beta-reduces a reified `functionof` callee and has nothing
/// to reduce for a bare operator.
///
/// This is what makes §06's two `pushfwd` spellings agree: a synthesized one-builtin
/// `f_inv` is emitted as the bare operator (what `broadcast` needs, and what a user writes
/// in `bijection(exp, log, …)`) while a composed one is a lambda that beta-reduces — both
/// land on the same direct builtin call.
fn builtin_callee_head(m: &Module, id: NodeId) -> Option<CallHead> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::User(callee) = c.head else {
        return None;
    };
    match m.node(callee) {
        Node::Const(sym) => Some(CallHead::Builtin(*sym)),
        _ => None,
    }
}

/// Re-allocate `id`'s call with `head`, keeping its positional/named arguments
/// (the callee child is dropped — it has become the head).
fn rebuild_with_head(m: &mut Module, id: NodeId, head: CallHead) -> NodeId {
    let Node::Call(c) = m.node(id) else {
        return id;
    };
    let call = flatppl_core::Call {
        head,
        args: c.args.clone(),
        named: c.named.clone(),
        inputs: c.inputs.clone(),
    };
    m.alloc(Node::Call(call))
}
