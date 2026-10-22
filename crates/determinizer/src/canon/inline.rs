//! Pass 2: inline residual user-defined function/kernel calls so no
//! `(%call User ...)` reaches consumers.
//!
//! A live `(%call (%ref self scale) 1.5)` is exactly what flatppl-js cannot
//! evaluate on the score path:
//! FlatPDL is defined as deterministic ops + the six
//! `builtin_*` primitives, and a call to a user function is neither. This
//! pass beta-reduces every such call away, because eliminating one is strictly
//! more useful than merely detecting it.
//!
//! `is_flatpdl` (`conformance.rs`) also rejects a residual `CallHead::User`
//! (`NonConformKind::ResidualUserCall`), so a call this pass CANNOT reduce
//! refuses at the driver's exit gate instead of reporting FlatPDL-conformant.
//! Two shapes reach it: an unresolved callee, and an arity mismatch at the call
//! site.

use std::collections::{HashMap, HashSet};

use flatppl_core::{Call, CallHead, Module, NamedKind, Node, NodeId, Symbol};

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
    rewrite_calls(m, true)
}

/// Resolve wrappers before density recognition without applying reified draw
/// boundaries early or pinning their provenance.
pub(crate) fn normalize_call_wrappers(m: &mut Module) -> bool {
    rewrite_calls(m, false)
}

fn rewrite_calls(m: &mut Module, reduce: bool) -> bool {
    let mut changed = false;
    let mut mapped = HashMap::new();
    let pairs: Vec<(flatppl_core::BindingId, NodeId)> =
        m.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    for (bid, root) in pairs {
        // `reduce_kernel_application` needs `&mut Module` and returns a fresh
        // body `NodeId`, so apply it via a manual bottom-up walk rather than
        // `map_tree` (whose closure only gets `&Module`, no `alloc`).
        let new = inline_walk(m, root, &mut mapped, None, reduce);
        if new != root {
            m.set_binding_rhs(bid, new);
            changed = true;
        }
    }
    changed
}

/// Specialize a callback body before its remaining live values are reified.
pub(crate) fn inline_expression(m: &mut Module, root: NodeId) -> NodeId {
    let promoted = super::promoted_input_names(m);
    inline_walk(m, root, &mut HashMap::new(), Some(&promoted), true)
}

/// Bottom-up: inline within children first (so a callee itself containing a
/// user call is reduced before we look at this node), then attempt to reduce
/// this node if it is a user call. Rebuilds via `rebuild_with_children` for
/// the child layer — the same rebuild `map_tree` uses, kept in one place so
/// both stay consistent in how a `Call`'s children decode back into
/// head/args/named.
fn inline_walk(
    m: &mut Module,
    id: NodeId,
    mapped: &mut HashMap<NodeId, NodeId>,
    projections: Option<&HashSet<Symbol>>,
    reduce: bool,
) -> NodeId {
    if let Some(&result) = mapped.get(&id) {
        return result;
    }
    let children: Vec<NodeId> = m.node(id).children();
    let mut any_child_changed = false;
    let new_children: Vec<NodeId> = children
        .iter()
        .map(|&c| {
            let nc = inline_walk(m, c, mapped, projections, reduce);
            any_child_changed |= nc != c;
            nc
        })
        .collect();
    let original = id;
    let mut id = if any_child_changed {
        rebuild_with_children(m, id, &new_children)
    } else {
        id
    };
    if let Some(promoted) = projections {
        id = super::project_tuple(m, id, promoted)
            .or_else(|| super::project(m, id, promoted))
            .unwrap_or(id);
    }
    // Then reduce this node if it is a user call.
    if let Node::Call(c) = m.node(id)
        && matches!(c.head, CallHead::User(_))
    {
        if let Some(head) = builtin_callee_head(m, id) {
            id = rebuild_with_head(m, id, head);
        } else if let Some(expanded) = expand_composed_call(m, id) {
            id = inline_walk(m, expanded, mapped, projections, reduce);
        } else if reduce && let Some(reduced) = reduce_kernel_application(m, id) {
            // The reduced body may itself contain further user calls
            // (e.g. a function whose body calls another function).
            id = inline_walk(m, reduced, mapped, projections, reduce);
        }
    }
    id = normalize_builtin_call(m, id).unwrap_or(id);
    mapped.insert(original, id);
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
    match m.node(crate::density::resolve_ref_chain(m, callee).0) {
        Node::Const(sym) => Some(CallHead::Builtin(*sym)),
        _ => None,
    }
}

/// Composition preserves the first interface; each later stage is one ordinary
/// application to the previous result. Lifting delegates to broadcast's cell call.
pub(crate) fn expand_composed_call(m: &mut Module, id: NodeId) -> Option<NodeId> {
    let Node::Call(call) = m.node(id) else {
        return None;
    };
    let CallHead::User(callee) = call.head else {
        return None;
    };
    let mut call = call.clone();
    let callee = crate::density::resolve_ref_chain(m, callee).0;
    let Node::Call(wrapper) = m.node(callee) else {
        return None;
    };
    let CallHead::Builtin(op) = wrapper.head else {
        return None;
    };
    let name = m.resolve(op).to_string();
    let components = wrapper.args.to_vec();
    let mut wrapper = wrapper.clone();
    let first = *components.first()?;
    match name.as_str() {
        "fchain" => {
            call.head = CallHead::User(first);
            let mut result = m.alloc(Node::Call(call));
            for &next in &components[1..] {
                result = m.alloc(Node::Call(Call {
                    head: CallHead::User(next),
                    args: Box::new([result]),
                    named: Box::default(),
                    inputs: None,
                }));
            }
            Some(result)
        }
        "bijection" => {
            call.head = CallHead::User(first);
            Some(m.alloc(Node::Call(call)))
        }
        "broadcasted" => {
            call.head = CallHead::Builtin(m.intern("broadcast"));
            call.args = std::iter::once(first)
                .chain(call.args.iter().copied())
                .collect();
            Some(m.alloc(Node::Call(call)))
        }
        "normalize" | "truncate" | "locscale" | "iid" | "relabel" | "pushfwd" | "weighted"
        | "logweighted"
            if matches!(m.type_of(callee), Some(flatppl_core::Type::Kernel { .. })) =>
        {
            let index = usize::from(matches!(
                name.as_str(),
                "pushfwd" | "weighted" | "logweighted"
            ));
            call.head = CallHead::User(*components.get(index)?);
            wrapper.args[index] = m.alloc(Node::Call(call));
            Some(m.alloc(Node::Call(wrapper)))
        }
        _ => None,
    }
}

/// Numeric backends receive positional operands in the catalogue's order.
/// Record/table construction and special operations have no ordinary named
/// signature and are left intact.
fn normalize_builtin_call(m: &mut Module, id: NodeId) -> Option<NodeId> {
    let Node::Call(call) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(op) = call.head else {
        return None;
    };
    if call.inputs.is_some() || call.named.iter().any(|arg| arg.kind != NamedKind::Kwarg) {
        return None;
    }
    let cat = flatppl_infer::builtin_catalogue();
    let name = m.resolve(op);
    if flatppl_infer::constructor_param_names(name).is_some() {
        return None;
    }
    let names = cat.base_param_names(name)?;
    let arity = cat.base_arity(name)?;
    if arity.max != Some(names.len()) {
        return None;
    }
    let mut call = call.clone();
    let names: Vec<_> = names.iter().map(|name| m.intern(name)).collect();
    let args = crate::kernel::bind_optional_call_args(
        m,
        &names,
        arity.min,
        &call.args,
        &call.named,
        false,
    )?;
    if call.named.is_empty() && call.args.as_ref() == args {
        return None;
    }
    call.args = args.into();
    call.named = Box::default();
    Some(m.alloc(Node::Call(call)))
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
