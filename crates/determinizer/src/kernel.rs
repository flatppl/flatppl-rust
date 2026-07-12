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
use flatppl_core::{
    Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Symbol,
};

/// A resolved kernel: its reified body and its boundary inputs as
/// `(name, body-target-ref)` pairs. For a `%specinputs` boundary the pairs are
/// in the authored (positional) order; for an `%autoinputs` boundary they are
/// the auto-traced `elementof` leaves in canonical (name-sorted) order.
/// `auto` distinguishes the two: an `%autoinputs` boundary is keyword-only
/// (spec §04 "no argument order can be inferred"), so a positional application
/// of it refuses.
pub(crate) struct Kernel {
    pub body: NodeId,
    pub inputs: Vec<(Symbol, Ref)>,
    pub auto: bool,
}

/// Read a reification's boundary inputs as `(name, body-target-ref)` pairs,
/// mirroring `infer`'s `input_entries` dual dispatch (`infer/src/ops.rs`): a
/// `%specinputs` boundary carries them inline; an `%autoinputs` (keyword-only)
/// boundary reads them from the module's auto-inputs side-table
/// ([`Module::auto_inputs_of`], filled by phase inference). Returns the inputs
/// and whether the boundary is `%autoinputs`. `None` for a reification with no
/// boundary, an empty boundary, or an `%autoinputs` boundary whose side-table
/// entry has not been filled (callers requiring exactly one input check the
/// length themselves).
fn boundary_inputs(
    m: &Module,
    reif_id: NodeId,
    inputs: &Option<Inputs>,
) -> Option<(Vec<(Symbol, Ref)>, bool)> {
    match inputs.as_ref()? {
        Inputs::Spec(entries) if !entries.is_empty() => {
            Some((entries.iter().map(|(nm, r)| (*nm, *r)).collect(), false))
        }
        Inputs::Spec(_) => None,
        Inputs::Auto => {
            let entries = m.auto_inputs_of(reif_id)?;
            if entries.is_empty() {
                return None;
            }
            Some((entries.iter().map(|(nm, r)| (*nm, *r)).collect(), true))
        }
    }
}

/// Resolve `k_arg` to a `kernelof(body, <boundary>)`. `None` for any
/// non-`kernelof` shape or a `kernelof` with no boundary inputs. The boundary
/// may be `%specinputs` (inline) OR `%autoinputs` (auto-traced, keyword-only) —
/// both are read via [`boundary_inputs`]. Returns ALL inputs; callers that
/// require exactly one check the length themselves.
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
    let (inputs, auto) = boundary_inputs(m, resolved, &c.inputs)?;
    Some(Kernel { body, inputs, auto })
}

/// Resolve `k_arg` to a reified callable — `kernelof` OR `functionof` — as a
/// `(body, boundary-inputs)` pair. `None` for any other shape, a call with
/// more than one positional argument, or a reification with no boundary inputs.
/// The boundary may be `%specinputs` (inline) OR `%autoinputs` (auto-traced,
/// keyword-only) — both are read via [`boundary_inputs`], and the resolved
/// `Kernel::auto` flag records which, so [`reduce_kernel_application`] can hold
/// an `%autoinputs` callable to keyword-only application. Returns ALL inputs;
/// callers that require exactly one check the length themselves.
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
    let (inputs, auto) = boundary_inputs(m, resolved, &c.inputs)?;
    Some(Kernel { body, inputs, auto })
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
/// Three application forms are recognized, distinguished structurally by the
/// application's own argument shape (not by which reifier produced `k` —
/// spec §04 does not tie the reifier name to the argument form):
/// - KEYWORD arguments (`k(name = value)`): each boundary input is bound BY
///   NAME to the supplied keyword of the same name. This is the only form an
///   `%autoinputs` (keyword-only) kernel supports (§04: "no argument order can
///   be inferred"), and a `%specinputs` kernel supports it too (§04: an
///   explicit boundary supports keyword args in addition to positional).
///   Binding is an exact bijection: every boundary input supplied once, and no
///   keyword without a matching boundary name — a missing or extra name refuses
///   (`None`), never leaving a boundary input free (a silent wrong density).
/// - a single `record(...)` argument: each boundary input is bound BY FIELD
///   NAME (the `k(record(mu = 1.5))` idiom — `record_field`).
/// - one or more POSITIONAL arguments: bound BY POSITION, arg\[i\] → the
///   i-th boundary entry (the `mk(0.0)` idiom). Positional binding is
///   `%specinputs`-ONLY: an `%autoinputs` kernel is keyword-only (§04), so a
///   positional application of one refuses rather than attach an argument to an
///   arbitrarily-ordered traced input. Arity must match the input count
///   exactly; a mismatch refuses (`None`) rather than guessing.
///
/// Note the record form binds BY FIELD NAME even when the kernel has exactly
/// one boundary input: `k(record(mu = 1.5))` looks up the input's own name as
/// a field of the record — it never binds the record as a whole positionally
/// to that single input. A field-name mismatch (the record lacks a field
/// matching the input's name) cleanly refuses (`None`) via `record_field`'s
/// `?`, rather than falling back to binding the whole record positionally.
///
/// An `%autoinputs` (keyword-only, boundary-less) reification IS handled: its
/// auto-traced boundary names + refs are read from the module's auto-inputs
/// side-table via [`boundary_inputs`] ([`Module::auto_inputs_of`]), so the
/// keyword form binds them by name and the positional form refuses.
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
    let args: Vec<NodeId> = c.args.to_vec();
    // Keyword arguments supplied at the application site (`k(name = value)`).
    let kwargs: Vec<(Symbol, NodeId)> = c
        .named
        .iter()
        .filter(|na| na.kind == NamedKind::Kwarg)
        .map(|na| (na.name, na.value))
        .collect();
    if args.is_empty() && kwargs.is_empty() {
        return None;
    }
    let kernel = resolve_reified(m, callee)?;

    let (resolved, _) = resolve_ref_one(m, kernel.body);
    let mut body = match draw_argument(m, resolved) {
        Some(law) => resolve_ref_one(m, law).0,
        None => resolved,
    };

    // KEYWORD application: bind each boundary input by name. The only form an
    // `%autoinputs` (keyword-only) kernel supports (§04); a `%specinputs` kernel
    // supports it too. Refuse a keyword/positional mix, or any bijection failure
    // (arity mismatch, or a boundary input with no matching keyword) rather than
    // leave a boundary input free — a silent wrong density.
    if !kwargs.is_empty() {
        if !args.is_empty() || kwargs.len() != kernel.inputs.len() {
            return None;
        }
        for (name, target) in &kernel.inputs {
            let value = kwargs.iter().find(|(n, _)| n == name).map(|(_, v)| *v)?;
            body = substitute_ref(m, body, target.name, value);
        }
        return Some(body);
    }

    if args.len() == 1 && is_record(m, args[0]) {
        for (name, target) in kernel.inputs {
            let value = record_field(m, args[0], name)?;
            body = substitute_ref(m, body, target.name, value);
        }
    } else if !kernel.auto && args.len() == kernel.inputs.len() {
        // POSITIONAL binding — `%specinputs`-only. An `%autoinputs` kernel is
        // keyword-only (§04), so a positional application of one falls through to
        // the refuse below rather than binding by an uninferable position.
        for (arg, (_, target)) in args.iter().zip(kernel.inputs.iter()) {
            body = substitute_ref(m, body, target.name, *arg);
        }
    } else {
        // Arity mismatch, or a positional application of a keyword-only
        // `%autoinputs` kernel — refuse rather than mis-lower.
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
