//! The nullary form of the measure-kernel evaluation primitives (spec §07).
//!
//! A measure is a nullary kernel, so `builtin_logdensityof(M, x)`,
//! `builtin_sample(rng, M, n, ...)` and the transports omit `kernel_input`.
//! The FlatPDL profile admits no measure-valued application, so a closed
//! built-in measure `K(args)` in the kernel slot becomes the kernel `K` and the
//! input `record(args)`: the same pair the density and sample lowerings emit
//! for `logdensityof(K(args), x)` and `draw(K(args))`. A box measure member has
//! no input and keeps the nullary form.

use std::collections::HashSet;

use flatppl_core::{Call, CallHead, Module, Node, NodeId};

use crate::density::{build_record, resolve_ref, split_kernel_constructor};

/// Rewrite every nullary primitive over a closed built-in measure. Returns
/// whether any call changed.
pub(crate) fn split_measure_kernels(m: &mut Module) -> bool {
    let mut changed = false;
    let mut malformed = HashSet::new();
    while let Some((bid, node)) = crate::driver::find_node(m, |id| {
        !malformed.contains(&id) && split_slot(m, id).is_some()
    }) {
        let slot = split_slot(m, node).expect("matched");
        let Node::Call(call) = m.node(node).clone() else {
            unreachable!("a primitive is a call")
        };
        // Records are no broadcast operands (spec §04), so a broadcast maps a
        // per-cell function of the plain primitive, which the next pass splits.
        if matches!(call.head, CallHead::Builtin(h) if m.resolve(h) == "broadcast") {
            let Node::Const(op) = *m.node(call.args[0]) else {
                unreachable!("a broadcast primitive head is a constant")
            };
            let Some(lowered) = crate::density::broadcast::lower_held_measure(m, node, op) else {
                malformed.insert(node);
                continue;
            };
            let rhs = crate::driver::substitute_in_tree(m, m.binding(bid).rhs, node, lowered);
            m.set_binding_rhs(bid, rhs);
            changed = true;
            continue;
        }
        let (resolved, _) = resolve_ref(m, call.args[slot]);
        // A malformed constructor call is left for the conformance check to report.
        let Some((ctor, fields)) = split_kernel_constructor(m, resolved) else {
            malformed.insert(node);
            continue;
        };
        let kernel = m.alloc(Node::Const(ctor));
        let input = build_record(m, &fields);
        let mut args = call.args[..slot].to_vec();
        args.extend([kernel, input]);
        args.extend_from_slice(&call.args[slot + 1..]);
        let rebuilt = m.alloc(Node::Call(Call {
            args: args.into(),
            ..call
        }));
        let rhs = crate::driver::substitute_in_tree(m, m.binding(bid).rhs, node, rebuilt);
        m.set_binding_rhs(bid, rhs);
        changed = true;
    }
    changed
}

/// The kernel slot of `id` when it is a measure-evaluation primitive, plain or
/// broadcast, whose kernel argument is a closed built-in measure.
fn split_slot(m: &Module, id: NodeId) -> Option<usize> {
    let Node::Call(call) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    let slot = match m.resolve(head) {
        // `broadcast(P, M, values)`. A broadcast sample has no lowering here.
        "broadcast" => match m.node(*call.args.first()?) {
            Node::Const(p) if m.resolve(*p) != "builtin_sample" && call.args.len() == 3 => {
                1 + flatppl_infer::builtins::kernel_tag_index(m.resolve(*p))?
            }
            _ => return None,
        },
        name => flatppl_infer::builtins::kernel_tag_index(name)?,
    };
    is_builtin_measure(m, *call.args.get(slot)?).then_some(slot)
}

/// Whether `tag` names a closed measure built by a built-in measure kernel.
fn is_builtin_measure(m: &Module, tag: NodeId) -> bool {
    let (resolved, _) = resolve_ref(m, tag);
    let Node::Call(call) = m.node(resolved) else {
        return false;
    };
    match call.head {
        CallHead::Builtin(sym) => flatppl_infer::constructor_param_names(m.resolve(sym)).is_some(),
        CallHead::User(callee) => crate::density::std_module_ctor_sym(m, callee).is_some(),
    }
}
