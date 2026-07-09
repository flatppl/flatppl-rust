//! Kernel resolution + application shared by the `kchain` marginal
//! (`marginal.rs`), the `jointchain` product (`jointchain.rs`), and
//! `density.rs`'s `lower_measure_density` reified-application dispatch.
//!
//! A `kernelof(body, %specinputs([(name, ref), …]))` reifies a measure `body`
//! with named boundary inputs. Each entry is `(name, Ref)`: `name` is what the
//! kernel input is called (matched to a prior variate field by auto-splat);
//! `Ref` is how the body references it — the SAME symbol as `name` for a
//! real-binding input (`(a (%ref self a))`), a placeholder (`(b (%ref %local
//! _b_))`) for an intermediate-variate input. Substitution replaces the `Ref`'s
//! symbol, so callers apply `substitute_ref(body, ref.name, value)`.
//!
//! `functionof(body, %specinputs(…))` over a measure-valued `body` is the same
//! reification under a different builtin name (spec §04 "Reification to
//! functions and kernels") — `resolve_reified` accepts both, but `resolve_kernel`
//! stays `kernelof`-only since `marginal.rs`/`jointchain.rs` depend on that.

use crate::density::{draw_argument, resolve_ref_one};
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

/// Resolve `k_arg` to a reified callable — `kernelof` OR `functionof` — as a
/// `(body, boundary-inputs)` pair. `None` for any other shape, a call with
/// more than one positional argument, or a reification without a
/// `%specinputs` boundary (an `%autoinputs`/keyword-only reification is not
/// handled here — see `reduce_kernel_application`'s doc comment). Returns ALL
/// inputs; callers that require exactly one check the length themselves.
pub(crate) fn resolve_reified(m: &Module, k_arg: NodeId) -> Option<Kernel> {
    let (resolved, _) = resolve_ref_one(m, k_arg);
    let Node::Call(c) = m.node(resolved) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    let head = m.resolve(sym);
    if (head != "kernelof" && head != "functionof") || c.args.len() != 1 {
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

/// If `node` is a reified-callable application `k(input)` / `k(a, b, …)`
/// where `k` resolves to a `kernelof(body, %specinputs(…))` OR a
/// `functionof(body, %specinputs(…))` over a measure-valued `body`
/// (`resolve_reified`), β-reduce it: substitute each boundary input's
/// body-ref with the bound argument, and return the reduced measure body.
/// `None` for any other shape.
///
/// Two application forms are recognized, distinguished structurally by the
/// application's own argument shape (not by which reifier produced `k` —
/// spec §04 does not tie the reifier name to the argument form):
/// - a single `record(...)` argument: each boundary input is bound BY FIELD
///   NAME (the `k(record(mu = 1.5))` idiom — `record_field`).
/// - one or more POSITIONAL arguments: bound BY POSITION, arg\[i\] → the
///   i-th `%specinputs` entry (the `mk(0.0)` idiom). Arity must match the
///   input count exactly; a mismatch refuses (`None`) rather than guessing.
///
/// An `%autoinputs` (keyword-only, boundary-less) reification is out of
/// scope here — `resolve_reified` already refuses it, since its traced input
/// order is inference metadata this module doesn't have access to.
///
/// `body` is commonly a bare `(%ref self x)` pointing at a `draw`-bound
/// stochastic value — the `x ~ Dist(...); k = kernelof(x, ...)` idiom (see
/// `fixtures/flatppl/minimal.flatppl`) — rather than an inline measure
/// expression. `substitute_ref` only rewrites literal descendants of its
/// root, so it cannot see through that ref into `x`'s own binding; resolve
/// one level of ref indirection and, if present, one level of `draw(...)`
/// unwrapping to reach the actual measure/law BEFORE substituting.
pub(crate) fn reduce_kernel_application(m: &mut Module, node: NodeId) -> Option<NodeId> {
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    let CallHead::User(callee) = c.head else {
        return None;
    };
    if c.args.is_empty() {
        return None;
    }
    let args: Vec<NodeId> = c.args.to_vec();
    let kernel = resolve_reified(m, callee)?;

    let (resolved, _) = resolve_ref_one(m, kernel.body);
    let mut body = match draw_argument(m, resolved) {
        Some(law) => resolve_ref_one(m, law).0,
        None => resolved,
    };
    if args.len() == 1 && is_record(m, args[0]) {
        for (name, target) in kernel.inputs {
            let value = record_field(m, args[0], name)?;
            body = substitute_ref(m, body, target.name, value);
        }
    } else if args.len() == kernel.inputs.len() {
        for (arg, (_, target)) in args.iter().zip(kernel.inputs.iter()) {
            body = substitute_ref(m, body, target.name, *arg);
        }
    } else {
        // Arity mismatch (more/fewer positional args than boundary inputs,
        // with no record to bind by name instead) — refuse rather than
        // mis-lower.
        return None;
    }
    Some(body)
}

/// Does `rec` (after one level of ref-resolution) denote a `record(...)`
/// call? Used to distinguish the by-field-name application form from the
/// positional form in `reduce_kernel_application`.
fn is_record(m: &Module, rec: NodeId) -> bool {
    let (resolved, _) = resolve_ref_one(m, rec);
    let Node::Call(c) = m.node(resolved) else {
        return false;
    };
    let CallHead::Builtin(sym) = c.head else {
        return false;
    };
    m.resolve(sym) == "record"
}

/// Look up field `name` in a `record(%field … )` node; `None` if `rec` is not
/// a record literal or lacks the field.
fn record_field(m: &Module, rec: NodeId, name: Symbol) -> Option<NodeId> {
    let (resolved, _) = resolve_ref_one(m, rec);
    let Node::Call(c) = m.node(resolved) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "record" {
        return None;
    }
    c.named.iter().find(|na| na.name == name).map(|na| na.value)
}
