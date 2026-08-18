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
    Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Scalar, Symbol,
    Type,
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
/// `root` with `new_id`. [`substitute_refs`] for a single name.
pub(crate) fn substitute_ref(m: &mut Module, root: NodeId, name: Symbol, new_id: NodeId) -> NodeId {
    substitute_refs(m, root, &[(name, new_id)])
}

/// Replace every `(%ref self n)` / `(%ref %local n)` in the subtree at `root` with
/// `map`'s value for `n`, for every entry of `map` SIMULTANEOUSLY. Append-only.
///
/// **Simultaneous, not one name after another, and that is the whole point of taking
/// a map.** Applying the entries in sequence captures a value the previous entry
/// inserted: for `k(z = w, w = 0.5)` over a body reading both, `z := w` writes `w`
/// into the body and the `w := 0.5` pass then rewrites the `w` it just inserted,
/// yielding `0.5` where §04 *Specifying reification boundaries* gives the ambient
/// `w` — the applied value is evaluated in the AMBIENT scope and is not itself part
/// of the reified graph, so nothing in it is a substitution target. One pass over
/// the ORIGINAL tree cannot capture, because a replaced node stands in wholesale and
/// is never descended into (the same property [`crate::driver::map_tree`] relies on).
///
/// Shadow-aware over ONE hazard, PER NAME: a nested `functionof`/`kernelof`
/// reification whose OWN boundary re-declares some of `map`'s names (see
/// [`shadows_name`]) is descended into with exactly those names dropped —
/// substituting them would rewrite references belonging to that reification's own
/// scope (variable capture). Beyond that one hazard this is still scope-UNAWARE:
/// sound under the workspace no-shadowing assumption for every other binding form (a
/// substituted symbol is never rebound by anything besides a reification boundary
/// inside the subtree).
pub(crate) fn substitute_refs(m: &mut Module, root: NodeId, map: &[(Symbol, NodeId)]) -> NodeId {
    if map.is_empty() {
        return root;
    }
    if let Node::Ref(Ref { ns, name }) = m.node(root) {
        if matches!(ns, RefNs::SelfMod | RefNs::Local) {
            if let Some((_, new_id)) = map.iter().find(|(n, _)| n == name) {
                return *new_id;
            }
        }
        return root;
    }
    // Per-name shadowing: a nested reification re-declaring SOME of the names blocks
    // those and no others.
    let active: Vec<(Symbol, NodeId)> = map
        .iter()
        .copied()
        .filter(|(name, _)| !shadows_name(m, root, *name))
        .collect();
    if active.is_empty() {
        return root;
    }
    let children: Vec<NodeId> = m.node(root).children();
    if children.is_empty() {
        return root;
    }
    let new_children: Vec<NodeId> = children
        .iter()
        .map(|&c| substitute_refs(m, c, &active))
        .collect();
    if new_children == children {
        return root;
    }
    crate::driver::rebuild_with_children(m, root, &new_children)
}

/// [`substitute_refs`] over the boundary entries `mode` admits, in ONE pass. See
/// [`Substitute`] for the mode split, and [`substitute_refs`] for why the pass must
/// cover every admitted entry at once rather than one entry at a time.
pub(crate) fn substitute_admitted(
    m: &mut Module,
    body: NodeId,
    bound: &[(Ref, NodeId)],
    mode: Substitute,
) -> NodeId {
    let map: Vec<(Symbol, NodeId)> = bound
        .iter()
        .filter(|(target, _)| match mode {
            Substitute::All => true,
            Substitute::LocalOnly => target.ns == RefNs::Local,
        })
        .map(|(target, value)| (target.name, *value))
        .collect();
    substitute_refs(m, body, &map)
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
/// - a single `record(...)` or `table(...)` argument: each boundary input is bound BY
///   FIELD/COLUMN NAME (the `k(record(mu = 1.5))` idiom — `is_splattable` /
///   `record_field`).
/// - one or more POSITIONAL arguments: bound BY POSITION, arg\[i\] → the
///   i-th boundary entry (the `mk(0.0)` idiom). Positional binding is
///   `%specinputs`-ONLY: an `%autoinputs` kernel is keyword-only (§04), so a
///   positional application of one refuses rather than attach an argument to an
///   arbitrarily-ordered traced input. Arity must match the input count
///   exactly; a mismatch refuses (`None`) rather than guessing.
///
/// Note the auto-splatting form binds BY NAME even when the kernel has exactly one
/// boundary input: `k(record(mu = 1.5))` looks up the input's own name as a field of
/// the record — it never binds the record as a whole positionally to that single
/// input. §04 "Calling conventions" is explicit that this is not a fallback: "A sole
/// positional record or table therefore always splats: whether its field or column
/// names match the callable's argument names decides only whether the call is valid,
/// never whether the splat occurs" (flatppl-design#74). So a name mismatch cleanly
/// refuses (`None`) via `record_field`'s `?`, rather than falling back to binding the
/// whole value positionally. `is_splattable` covers `table` for the same reason — see
/// its comment for why flatppl-design#78's single-input exemption cannot reach here.
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
    reduce_kernel_application_bound(m, node, Substitute::All).map(|app| app.body)
}

/// A β-reduced reified-callable application: the reduced body plus the boundary
/// binding that produced it, as `(body-target-ref, applied value)` pairs in
/// boundary order.
///
/// The pairs matter because [`substitute_ref`] is SYNTACTIC: it rewrites only
/// literal descendants of `body`, so a boundary reference reached through a
/// module binding (`mu2 = 2.0 * z` with `z` the boundary, or a `record` field
/// that is a `(%ref self b1)`) survives the reduction. A caller that goes on to
/// LOWER the reduced body must finish the substitution over what it emits —
/// `density::substitute_applied_boundary` does — or it scores a function of the
/// very parameter the application pinned.
pub(crate) struct AppliedReification {
    pub body: NodeId,
    pub bound: Vec<(Ref, NodeId)>,
}

/// Which boundary entries [`reduce_kernel_application_bound`] substitutes into the
/// body itself.
///
/// The distinction exists because substituting the SAME entry twice is a wrong
/// number, not a no-op: for `K(z = z + 1.0)` the syntactic pass writes `z + 1.0`
/// into the body, and a second pass over the result cannot tell its own output from
/// source, so it produces `z + 1.0 + 1.0`. A caller that finishes the substitution
/// over the emitted density must therefore NOT have it done here as well.
///
/// Whichever entries a mode admits are substituted in ONE simultaneous pass
/// ([`substitute_refs`]), so a cross-named applied value (`k(z = w, w = 0.5)`) is
/// never captured by the sibling entry that named it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Substitute {
    /// Every entry. For a caller that performs no further substitution of its own —
    /// `canon::inline`, the sampler, and the change-of-variables sites, none of which
    /// emit a density to finish over.
    All,
    /// `%local` placeholder targets ONLY, leaving every same-module target to the
    /// caller's own finish.
    ///
    /// The split is exactly the split of what a later pass CAN reach. A same-module
    /// target is a module binding, so `density::substitute_applied_boundary` reaches
    /// every occurrence of it — the literal ones in the body included — and reaches
    /// them through bindings the syntactic walk stops at. A `%local` placeholder has
    /// no module binding to reach it through, and §04 *Placeholders and holes*
    /// requires it to appear inside the reified expression, so it must be bound here
    /// or not at all.
    ///
    /// The two sets are disjoint, so nothing is substituted twice. They also exhaust
    /// what [`substitute_ref`] can rewrite — it matches `RefNs::SelfMod | RefNs::Local`
    /// and nothing else — so nothing this mode declines is left unbound. That is a
    /// statement about those two namespaces only: a `RefNs::Module(alias)` boundary
    /// target would be substituted by NEITHER pass, and it is unreachable here for a
    /// different reason, namely that [`resolve_reified`] admits only a local
    /// `kernelof`/`functionof` node, whose boundary entries name nodes in its own
    /// module. A future cross-module boundary would need its own handling, not this
    /// split widened.
    LocalOnly,
}

/// [`reduce_kernel_application`] keeping the boundary binding it applied, and
/// substituting only what `mode` admits. See [`AppliedReification`] for why a
/// lowering caller needs the binding, and [`Substitute`] for why it must not also
/// have the substitution done here.
pub(crate) fn reduce_kernel_application_bound(
    m: &mut Module,
    node: NodeId,
    mode: Substitute,
) -> Option<AppliedReification> {
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
    // The whole binding is computed BEFORE anything is substituted, so the
    // substitution can run as one simultaneous pass ([`substitute_refs`]) rather than
    // one pass per entry — which captures a sibling's applied value.
    let mut bound: Vec<(Ref, NodeId)> = Vec::with_capacity(kernel.inputs.len());
    if !kwargs.is_empty() {
        if !args.is_empty() || kwargs.len() != kernel.inputs.len() {
            return None;
        }
        for (name, target) in &kernel.inputs {
            let value = kwargs.iter().find(|(n, _)| n == name).map(|(_, v)| *v)?;
            bound.push((*target, value));
        }
    } else if args.len() == 1 && is_splattable(m, args[0]) {
        for (name, target) in &kernel.inputs {
            let value = record_field(m, args[0], *name)?;
            bound.push((*target, value));
        }
    } else if !kernel.auto && args.len() == kernel.inputs.len() {
        // POSITIONAL binding — `%specinputs`-only. An `%autoinputs` kernel is
        // keyword-only (§04), so a positional application of one falls through to
        // the refuse below rather than binding by an uninferable position.
        for (arg, (_, target)) in args.iter().zip(kernel.inputs.iter()) {
            bound.push((*target, *arg));
        }
    } else {
        // Arity mismatch, or a positional application of a keyword-only
        // `%autoinputs` kernel — refuse rather than mis-lower.
        return None;
    }
    body = substitute_admitted(m, body, &bound, mode);
    Some(AppliedReification { body, bound })
}

/// Does `rec` (after one level of ref-resolution) denote a `record(...)` or `table(...)`
/// call? Used to distinguish the by-name auto-splatting application form from the
/// positional one in [`reduce_kernel_application`].
///
/// §04 "Calling conventions" names BOTH: "`f(record(a = x, b = y, ...))` and
/// `f(table(a = x, b = y, ...))` are equivalent to `f(a = x, b = y, ...)`", and a sole
/// positional one of either "therefore always splats". Recognising only `record` sent a
/// sole positional TABLE down the positional arm, where a single-input reification bound
/// the whole table to that input — the whole-value reading §04 now rules out.
///
/// **The §04 amendment under review does not exempt this site.** flatppl-design#78 (OPEN,
/// owner-accepted, PENDING review) exempts "a callable with exactly one input whose
/// documented domain admits records or tables", so that `sum(t)` and `lengthof(t)` reduce
/// over the table. Its test is the CALLEE's arity and DOCUMENTED domain. Every callee
/// reaching here is a user `functionof`/`kernelof` reification ([`resolve_reified`] off a
/// `CallHead::User`), which has no documented domain at all — §07's "Domains" column
/// covers built-ins — so the exemption cannot apply however many inputs it declares. A
/// bare-builtin callee never reaches this function: `canon` rewrites it to a direct
/// builtin call, and the two `pushfwd` sites screen it off beforehand.
pub(crate) fn is_splattable(m: &Module, rec: NodeId) -> bool {
    matches!(splat_head(m, rec), Some("record") | Some("table")) || table_columns(m, rec).is_some()
}

/// The column names of an OPAQUE table — one with no syntactic `table(...)` head, such as a
/// `load_data` result — read from its INFERRED TYPE. `None` for anything that is not a
/// table, including a table whose type has not been inferred.
///
/// §13 `sec:determinization-signature` makes a `load_data`'s shape come from its declared
/// `valueset`, so the columns of `load_data("x.csv", cartpow(cartprod(a = reals, b = reals),
/// 4))` are statically known — they are right there in
/// `(%table (%columns (a …) (b …)) (%nrows 4))`. That is what lets §04's splat apply to an
/// opaque table exactly as to a literal one: the names the splat binds by do not depend on
/// the file's contents. `determinize` re-infers before lowering, so the type is populated by
/// the time this runs.
///
/// Type-based rather than syntax-based, so it stays permissive where the type is not known:
/// a `%deferred` or absent type gives `None`, the value is not treated as splattable, and
/// nothing is refused on a guess.
fn table_columns(m: &Module, rec: NodeId) -> Option<Vec<Symbol>> {
    let (resolved, _) = resolve_ref_one(m, rec);
    let ty = m.type_of(rec).or_else(|| m.type_of(resolved))?;
    match ty {
        Type::Table { columns, .. } => Some(columns.iter().map(|(n, _)| *n).collect()),
        _ => None,
    }
}

/// The `record`/`table` head of `rec` after one level of ref-resolution, or `None`.
fn splat_head(m: &Module, rec: NodeId) -> Option<&str> {
    let (resolved, _) = resolve_ref_one(m, rec);
    let Node::Call(c) = m.node(resolved) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    Some(m.resolve(sym))
}

/// The value §04's splat binds to `name`: field/column `name` of a `record(%field … )` or
/// `table(%field … )` literal, or — for an OPAQUE table — a synthesized column access.
/// `None` if `rec` is neither, or lacks the name.
///
/// `None` on a missing name is what makes a mismatch REFUSE rather than fall back to
/// whole-value binding.
///
/// **The opaque case.** A `load_data` result has no syntactic columns, so D6 could not
/// destructure it and the reduction refused — which left a §04-legal model
/// (`g(t)` against a `g` whose parameters are `t`'s column names) refusing with the generic
/// "residual user call", while its `table(...)` literal twin lowered. The columns are
/// statically known from the type ([`table_columns`]), and §03 "Tables" already spells the
/// per-column value: "Column access by name: `t.a` … returns the column with that name as a
/// vector", which the parser lowers to `get(t, "a")`. So the splat IS constructible — it is
/// the same set of column values the literal case binds, reached through `get` instead of
/// through the literal's `%field` list. Synthesizing it makes the two forms agree, which is
/// what §04 requires: "whether its field or column names match the callable's argument names
/// decides only whether the call is valid, never whether the splat occurs."
pub(crate) fn record_field(m: &mut Module, rec: NodeId, name: Symbol) -> Option<NodeId> {
    if matches!(splat_head(m, rec), Some("record") | Some("table")) {
        let (resolved, _) = resolve_ref_one(m, rec);
        let Node::Call(c) = m.node(resolved) else {
            return None;
        };
        return c.named.iter().find(|na| na.name == name).map(|na| na.value);
    }
    // Opaque table: bind the column access §03 defines, if the type declares that column.
    //
    // A multi-column splat that refuses on a LATER column leaves the `get` nodes already
    // allocated for the earlier ones unreferenced. Benign: the arena is append-only and never
    // freed per node (`crates/core/src/id.rs`), nothing reaches them, and inference rejects a
    // name mismatch before the determiniser runs anyway — so the partial case is only reachable
    // when some other layer has already failed.
    if !table_columns(m, rec)?.contains(&name) {
        return None;
    }
    let key = m.alloc(Node::Lit(Scalar::Str(m.resolve(name).into())));
    let get = m.intern("get");
    Some(m.alloc(Node::Call(Call {
        head: CallHead::Builtin(get),
        args: vec![rec, key].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    })))
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
