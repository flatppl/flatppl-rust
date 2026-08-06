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

/// How a node stands with respect to unwrapping a reification wrapper.
/// [`classify_reification`] computes it; a caller that only wants the body uses
/// [`resolve_closed_reification`], and a caller that must explain a refusal reads
/// the other variants so it names the actual obstacle.
pub(crate) enum Reification {
    /// A CLOSED reification, unwrapped to a fixpoint: the innermost body, which
    /// IS the reification's value. See [`classify_reification`] for the spec basis.
    Closed(NodeId),
    /// A NON-EMPTY boundary. Reaching the body needs a value bound to each input,
    /// which is §04 kernel-boundary semantics rather than an unwrapping.
    Parameterised,
    /// An empty boundary, but the body holds a free `%local` placeholder that no
    /// boundary declares. §04 *Placeholders and holes* forbids the module — "All
    /// placeholders must appear both in the expression to be reified and the
    /// boundary input keyword arguments" — so unwrapping here would emit a
    /// dangling `(%ref %local …)`. The front door for the shape is `infer`'s
    /// undeclared-placeholder error (`infer/src/ops.rs`, `reification_type`);
    /// this stays the conservative default for a pre-inference caller.
    FreePlaceholder,
    /// An `%autoinputs` boundary whose side-table entry phase inference has not
    /// filled ([`Module::auto_inputs_of`] `None`): UNKNOWN inputs, not zero.
    ///
    /// Not reachable through [`crate::determinize`] as things stand: the driver
    /// re-runs inference at the top of every reduction pass, which fills the
    /// side-table for every `%autoinputs` node — including one grafted in after the
    /// host's own inference. Kept as the conservative default for any caller that
    /// classifies before inference, since the alternative is to read an unfilled
    /// table as "no inputs" and unwrap a parameterised reification. Pinned by
    /// `classify_reification_reports_an_unfilled_boundary_rather_than_closed`, which
    /// tests the classification directly for that reason.
    Unfilled,
    /// Not a one-argument `functionof` / `kernelof` at all.
    Plain,
}

/// Classify `id` as a reification wrapper, unwrapping CLOSED layers to a fixpoint.
///
/// A closed reification — boundary declared and EMPTY — takes no inputs, so every
/// ancestor is closed over (§04 *Reification to functions and kernels*: the traced
/// `elementof` leaves "become the inputs of the reified callable", and there are
/// none). Two spec anchors say such a wrapper means its body, one per case:
///
/// - A closed reification of a MEASURE is a measure, by §06 *Uniform kernel
///   extension*: "Mathematically, a measure is equivalent to a transition kernel
///   with an empty first argument. So in FlatPPL, we unify measures and kernels and
///   identify measures with nullary kernels." The same section blesses scoring it —
///   `logdensityof` and friends "require closed measures (i.e. nullary kernels) as
///   inputs". That identity is what lets a caller route the wrapper as its body. It
///   says nothing about `lawof`'s own argument class, which §04 states separately
///   ("`lawof` reifies a value node"), so no caller may read it as licensing a
///   `lawof` over a measure.
/// - A closed reification of a FUNCTION has no such identity. There §04's
///   prohibition carries it, by its rationale clause: "No callables may have
///   nullary inputs, as this would make them equivalent to known values." Read
///   strictly that makes the wrapper ill-formed rather than defining it, but the
///   clause is the spec's own reading of what a nullary callable would be, and no
///   spec text contradicts it. The `pushfwd` forward-map sites rest on this.
///
/// Unwrapping is therefore a spelling change, not a semantic one, and it repeats:
/// §04's rationale applies at every level, so `functionof(functionof(f))` unwraps
/// twice. The loop stops at the first layer that is not closed and returns the body
/// reached — a parameterised lambda is a legitimate stopping point, and the value
/// the caller should then recognise.
pub(crate) fn classify_reification(m: &Module, id: NodeId) -> Reification {
    let mut cur = id;
    let mut body_reached: Option<NodeId> = None;
    // The IR is a DAG (§04), but a malformed FlatPIR could cycle; a hang is worse
    // than a refusal, so stop on a repeat rather than trust the input.
    let mut seen: Vec<NodeId> = Vec::new();
    loop {
        if seen.contains(&cur) {
            return Reification::Plain;
        }
        seen.push(cur);
        match classify_one(m, cur) {
            Reification::Closed(body) => {
                let (next, _) = resolve_ref_one(m, body);
                body_reached = Some(next);
                cur = next;
            }
            // Below at least one closed layer, the body we reached IS the value,
            // whatever this layer turned out to be.
            stopped => {
                return match body_reached {
                    Some(body) => Reification::Closed(body),
                    None => stopped,
                };
            }
        }
    }
}

/// One layer of [`classify_reification`], no unwrapping of the body.
fn classify_one(m: &Module, id: NodeId) -> Reification {
    let Node::Call(c) = m.node(id) else {
        return Reification::Plain;
    };
    let CallHead::Builtin(sym) = c.head else {
        return Reification::Plain;
    };
    let head = m.resolve(sym);
    if (head != "functionof" && head != "kernelof") || c.args.len() != 1 {
        return Reification::Plain;
    }
    let Some(inputs) = c.inputs.as_ref() else {
        return Reification::Plain;
    };
    let empty_boundary = match inputs {
        // An empty `Spec` boundary is unreachable by construction: both FlatPIR front
        // ends reject an empty input list ("a reification input list cannot be empty
        // (callables cannot be nullary)" — `flatpir::reader::parse_input_entries`, and
        // the same check in `flatpir::json`), and inference never emits `Spec` at all.
        // Handled anyway so an authored boundary and a traced one agree here.
        Inputs::Spec(entries) => entries.is_empty(),
        Inputs::Auto => match m.auto_inputs_of(id) {
            Some(entries) => entries.is_empty(),
            None => return Reification::Unfilled,
        },
    };
    if !empty_boundary {
        return Reification::Parameterised;
    }
    if has_free_local(m, c.args[0]) {
        return Reification::FreePlaceholder;
    }
    Reification::Closed(c.args[0])
}

/// The body of a CLOSED reification at `id`, unwrapped to a fixpoint; `None` for
/// every other shape. Thin wrapper over [`classify_reification`] for the sites that
/// only route and never explain.
pub(crate) fn resolve_closed_reification(m: &Module, id: NodeId) -> Option<NodeId> {
    match classify_reification(m, id) {
        Reification::Closed(body) => Some(body),
        _ => None,
    }
}

/// True iff the subtree at `root` refers to a `%local` placeholder that no
/// reification boundary WITHIN the subtree declares. Such a ref is bound by an
/// enclosing boundary, so unwrapping that boundary would leave it dangling.
///
/// This keeps the unwrap in [`classify_reification`] from emitting a dangling
/// `(%ref %local …)`, and `density::lower_reified_measure` screens its own
/// `%autoinputs` boundary with it (that path's `%specinputs` screen covers only
/// placeholder ENTRIES). The front door for the shape is now `infer`'s
/// undeclared-placeholder error (`infer/src/ops.rs`, `reification_type`), which
/// rejects `functionof(Normal(mu = _v_, sigma = 1.0))` before the determiniser
/// runs; these two screens stay as the backstop for a caller that ignores
/// inference's diagnostics.
pub(crate) fn has_free_local(m: &Module, root: NodeId) -> bool {
    fn walk(m: &Module, id: NodeId, bound: &mut Vec<Symbol>) -> bool {
        if let Node::Ref(Ref {
            ns: RefNs::Local,
            name,
        }) = m.node(id)
        {
            return !bound.contains(name);
        }
        let declared: Vec<Symbol> = match m.node(id) {
            Node::Call(c) => match c.inputs.as_ref() {
                Some(Inputs::Spec(entries)) => entries
                    .iter()
                    .filter(|(_, r)| r.ns == RefNs::Local)
                    .map(|(_, r)| r.name)
                    .collect(),
                Some(Inputs::Auto) => m
                    .auto_inputs_of(id)
                    .unwrap_or_default()
                    .iter()
                    .filter(|(_, r)| r.ns == RefNs::Local)
                    .map(|(_, r)| r.name)
                    .collect(),
                None => Vec::new(),
            },
            _ => Vec::new(),
        };
        let depth = bound.len();
        bound.extend(declared);
        let found = m.node(id).children().into_iter().any(|c| walk(m, c, bound));
        bound.truncate(depth);
        found
    }
    walk(m, root, &mut Vec::new())
}

/// Replace every `(%ref self name)` / `(%ref %local name)` in the subtree at
/// `root` with `new_id`. Append-only. Shadow-aware over ONE hazard: a nested
/// `functionof`/`kernelof` reification whose OWN boundary re-declares `name`
/// as one of its inputs (see [`shadows_name`]) is left untouched — descending
/// into it would rewrite a reference that belongs to that reification's OWN
/// scope, not the outer substitution (variable capture). Beyond that one
/// hazard this is still scope-UNAWARE: sound under the workspace no-shadowing
/// assumption for every other binding form (a substituted symbol is never
/// rebound by anything besides a reification boundary inside the subtree).
pub(crate) fn substitute_ref(m: &mut Module, root: NodeId, name: Symbol, new_id: NodeId) -> NodeId {
    if let Node::Ref(Ref { ns, name: rname }) = m.node(root) {
        if matches!(ns, RefNs::SelfMod | RefNs::Local) && *rname == name {
            return new_id;
        }
    }
    if shadows_name(m, root, name) {
        return root;
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

/// True iff `id` is a reification (`functionof`/`kernelof`) whose OWN
/// boundary declares `name` as one of its inputs' body-target refs — i.e.
/// `id`'s body re-binds `name` for itself. Checked via the node's `Inputs`:
/// an explicit `%specinputs` list inline, or an `%autoinputs` boundary via the
/// module's auto-inputs side-table ([`Module::auto_inputs_of`], filled by
/// phase inference). Exists because the same synthesized placeholder name
/// (commonly `_x_`, minted uniformly by `flatppl-syntax`'s single-arg lambda
/// lowering, `lower_lambda`) is reused across UNRELATED reifications rather
/// than gensym'd fresh per occurrence; two of them can end up nested (an
/// outer reification's body containing an inert, uninvoked inner one) with
/// the SAME boundary name, and [`substitute_ref`] must stop at the inner
/// one's edge rather than rewrite a reference that belongs to its own scope.
fn shadows_name(m: &Module, id: NodeId, name: Symbol) -> bool {
    let Node::Call(c) = m.node(id) else {
        return false;
    };
    match &c.inputs {
        Some(Inputs::Spec(entries)) => entries.iter().any(|(_, r)| r.name == name),
        Some(Inputs::Auto) => m
            .auto_inputs_of(id)
            .is_some_and(|entries| entries.iter().any(|(_, r)| r.name == name)),
        None => false,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use flatppl_core::Call;

    /// An `%autoinputs` boundary whose side-table entry is absent means the inputs are
    /// UNKNOWN, not none. Reading it as none would unwrap a reification that may well
    /// have boundary inputs, dropping them silently. Tested at the classification level
    /// because `determinize` re-infers before lowering, so no module reaches the
    /// corresponding refusal — see [`Reification::Unfilled`].
    #[test]
    fn classify_reification_reports_an_unfilled_boundary_rather_than_closed() {
        let src = "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
                   F = functionof(Normal(mu = a, sigma = 1.0))\n\
                   lp = logdensityof(lawof(F), 0.5)";
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);

        let f_sym = m.intern("F");
        let f_bid = m.binding_by_name(f_sym).expect("F is bound");
        let filled = m.binding(f_bid).rhs;
        // Inference filled this one, and it is genuinely closed.
        assert!(
            matches!(classify_reification(&m, filled), Reification::Closed(_)),
            "a traced boundary with no `elementof` leaves is closed"
        );

        // The same reification as a FRESH node: identical shape, no side-table entry.
        let (head, args) = match m.node(filled) {
            Node::Call(c) => (c.head, c.args.clone()),
            _ => unreachable!("F's RHS is a call"),
        };
        let unfilled = m.alloc(Node::Call(Call {
            head,
            args,
            named: Box::new([]),
            inputs: Some(Inputs::Auto),
        }));
        assert!(
            m.auto_inputs_of(unfilled).is_none(),
            "a fresh node has no side-table entry"
        );
        assert!(
            matches!(classify_reification(&m, unfilled), Reification::Unfilled),
            "an unfilled boundary is UNKNOWN, never closed"
        );
        assert!(
            resolve_closed_reification(&m, unfilled).is_none(),
            "and it must not be unwrapped"
        );
    }
}
