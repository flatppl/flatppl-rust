//! Pass 2: inline residual user-defined function/kernel calls so no
//! `(%call User ...)` reaches consumers (Buffy #263; unblocks #261).
//!
//! `is_flatpdl` (`conformance.rs`) never flagged a residual `CallHead::User`
//! call — it only rejects Measure/Likelihood-typed nodes, stray Kernel types,
//! and `Stochastic` phase — so a determinized module could carry a live
//! `(%call (%ref self scale) 1.5)` and still report FlatPDL-conformant. That
//! residual is exactly what flatppl-js cannot evaluate on the score path
//! (Buffy #261): FlatPDL is defined as deterministic ops + the six
//! `builtin_*` primitives, and a call to a user function is neither. This
//! pass beta-reduces every such call away rather than widening
//! `is_flatpdl`'s reject list, because a residual call CAN always be
//! inlined here (unlike a genuinely un-lowerable measure op) — eliminating
//! it is strictly more useful than merely detecting it.

use flatppl_core::{CallHead, Module, Node, NodeId};

use crate::driver::rebuild_with_children;
use crate::kernel::reduce_kernel_application;

/// Replace each `(%call User(callee) args)` with its beta-reduced body. Reuses
/// `reduce_kernel_application` (which resolves the reified callee, binds inputs
/// by position/keyword/record-splat, and substitutes via `substitute_ref`). A
/// call it cannot reduce (unresolved callee, arity mismatch) is left in place
/// — refuse-free; `is_flatpdl` never flagged this shape either way, so leaving
/// one in place changes nothing about conformance, only about whether a
/// consuming engine can evaluate it.
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
            if let Some(reduced) = reduce_kernel_application(m, id) {
                // The reduced body may itself contain further user calls
                // (e.g. a function whose body calls another function).
                return inline_walk(m, reduced);
            }
        }
    }
    id
}
