//! Greedy legalizer driver: repeatedly scans for measure-layer nodes, applies the
//! first matching rewrite rule, and loops until the module is FlatPDL-conformant
//! or no rule applies (refusal).
//!
//! **Task 2 rules implemented here:**
//! - β-law: `lawof(draw ?m)` → `?m`   (spec §04 identity)
//! - refuse-everything-else: any other measure-layer construct → `RefuseError`
//!
//! Task 3+ will add `logdensityof` lowering and combinator rules; the dispatch
//! table (`apply_rule`) is the single extension point.

use crate::refuse::RefuseError;
use flatppl_core::{BindingId, CallHead, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol};

/// The measure-algebra vocabulary: op names whose presence signals a node that
/// must be eliminated before the module is FlatPDL-conformant. This list matches
/// the spec §04 / §06 measure-layer operations listed in the task brief.
const MEASURE_VOCAB: &[&str] = &[
    "draw",
    "lawof",
    "kernelof",
    "joint",
    "iid",
    "jointchain",
    "kchain",
    "markovchain",
    "kscan",
    "superpose",
    "weighted",
    "logweighted",
    "normalize",
    "truncate",
    "pushfwd",
    "locscale",
    "bayesupdate",
    "disintegrate",
    "restrict",
    "logdensityof",
    "densityof",
    "rand",
    // `totalmass` is OUT of FlatPDL: it is a query op that takes a measure, which
    // is no longer a value (flatpdl-determinise.md §1–§2). It must be eliminated
    // (refused, in this MVP) like any other measure-layer op — never emitted.
    "totalmass",
    "likelihoodof",
    "joint_likelihood",
];

/// Transform `m` into a FlatPDL-conformant module, or return the first construct
/// that cannot be legalized.
///
/// Works on a clone of `m`; the original is not modified.
pub fn determinize(m: &Module) -> Result<Module, RefuseError> {
    let mut work = m.clone();

    loop {
        // Re-run inference (idempotent) so type / phase tables are fresh.
        let _ = flatppl_infer::infer(&mut work);

        // Check for measure-layer nodes.
        let target = find_measure_node(&work);

        match target {
            None => {
                // No measure-layer nodes remain.
                match crate::is_flatpdl(&work) {
                    Ok(()) => return Ok(work),
                    Err(violations) => {
                        // Residual conformance issue not covered by the
                        // measure-vocab scan (e.g. a stochastic-phase Ref that
                        // slipped through). Report the first violation as a
                        // refusal so the caller gets a clear signal.
                        let v = &violations[0];
                        return Err(RefuseError {
                            node: v.node,
                            construct: format!("{:?}", v.kind),
                            reason: v.reason.clone(),
                        });
                    }
                }
            }
            Some((bid, node_id)) => {
                apply_rule(&mut work, bid, node_id)?;
                // Loop: re-scan after the rewrite.
            }
        }
    }
}

/// Scan bindings for the next measure-layer node to reduce. Returns
/// `(binding_id, node_id)` of the chosen node.
///
/// **Two-pass, query-first.** A `logdensityof` query pins the latent `draw`s it
/// scores (it rewrites their bindings to the scored value), so it must fire
/// *before* those bare `draw` bindings are reached and refused: a draw consumed
/// by a density query is reducible *through* that query, not on its own. So we
/// first look for any `logdensityof` node (scanning bindings in source order,
/// outermost-first), and only if none exists fall back to the general
/// outermost-measure scan (β-law `lawof`, or a refusal target). Without this,
/// the source-order scan would hit a `draw` binding first and refuse before the
/// query that would have legalised it.
fn find_measure_node(m: &Module) -> Option<(BindingId, NodeId)> {
    if let Some(hit) = find_op_node(m, "logdensityof") {
        return Some(hit);
    }
    for (bid, binding) in m.bindings() {
        if let Some(id) = find_in_subtree(m, binding.rhs) {
            return Some((bid, id));
        }
    }
    None
}

/// Find the first node (outermost, BFS) whose builtin head is named `op`,
/// scanning bindings in source order.
fn find_op_node(m: &Module, op: &str) -> Option<(BindingId, NodeId)> {
    for (bid, binding) in m.bindings() {
        let mut queue = vec![binding.rhs];
        let mut qi = 0;
        while qi < queue.len() {
            let id = queue[qi];
            qi += 1;
            if let Node::Call(c) = m.node(id) {
                if let CallHead::Builtin(sym) = c.head {
                    if m.resolve(sym) == op {
                        return Some((bid, id));
                    }
                }
            }
            m.for_each_child(id, |child| queue.push(child));
        }
    }
    None
}

/// Walk the subtree rooted at `root`, returning the outermost measure-layer
/// `NodeId` (BFS-order, so the root itself wins over its children).
fn find_in_subtree(m: &Module, root: NodeId) -> Option<NodeId> {
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        if is_measure_layer(m, id) {
            return Some(id);
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    None
}

/// A node is in the measure layer if it is a `Call` whose builtin head is in
/// `MEASURE_VOCAB`. Distribution constructors (`Normal`, `Bernoulli`, …) produce
/// `Measure`-typed values but are *not* measure-algebra operations — they are
/// primitive data constructors that remain as arguments to `draw` after β-law, or
/// as arguments to `logdensityof`/`densityof` in Task 3+. Scanning by op name
/// (not type) ensures we target algebra operations, not their primitive operands.
fn is_measure_layer(m: &Module, id: NodeId) -> bool {
    if let Node::Call(c) = m.node(id) {
        if let CallHead::Builtin(sym) = c.head {
            let name = m.resolve(sym);
            if MEASURE_VOCAB.contains(&name) {
                return true;
            }
        }
    }
    false
}

/// Attempt to rewrite `target_node` (inside `bid`'s subtree) using the first
/// matching rule. Returns `Ok(())` on a successful rewrite; `Err(RefuseError)`
/// when no rule applies.
///
/// After a successful rewrite the module's node arena has grown by at most one
/// node (β-law replaces a `lawof(draw ?m)` with a reference to `?m`'s node —
/// zero new nodes since we reuse the existing `?m` NodeId). Each rewrite
/// strictly lowers the count of measure-layer nodes, guaranteeing termination.
fn apply_rule(m: &mut Module, bid: BindingId, target_node: NodeId) -> Result<(), RefuseError> {
    // --- density disintegration: logdensityof(lawof(M), v) → deterministic density ---
    if is_op(m, target_node, "logdensityof") {
        let new_root = crate::density::lower_logdensityof(m, target_node)?;
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, new_root);
        m.set_binding_rhs(bid, new_rhs);
        // After lowering a logdensityof query, some measure bindings (e.g.
        // `m = weighted(...)`) may now be dead code — the draw was pinned and
        // the combinator binding is no longer referenced.  Sweep them out now so
        // the outer scan loop does not encounter them as unhandled measure-layer
        // nodes on the next iteration.
        sweep_dead_measure_bindings(m);
        return Ok(());
    }

    // --- rand / builtin_sample: sampling-side slice — deferred ---
    // `rand(rng, M)` threads an RNG through the measure algebra and returns a
    // (value, new_rng) tuple (spec §07). The determiniser MVP lowers the density
    // side only; sampling/`rand` is a later slice whose implementation requires
    // RNG-threading and shared-ancestor preservation. Refuse immediately with a
    // clear message rather than falling through to the generic fallback.
    if is_op(m, target_node, "rand") {
        return Err(RefuseError {
            node: target_node,
            construct: "rand".to_string(),
            reason: "sampling/`rand` is a later slice; this MVP lowers density only — \
                     deferred to the sample-side determinizer (spec §07 / RNG-threading)"
                .to_string(),
        });
    }

    // --- β-law: lawof(draw ?m) → ?m ---
    if let Some(measure_id) = try_beta_law(m, target_node) {
        // Replace every occurrence of `target_node` in the binding tree.
        // Because we scan outermost-first the target IS the binding's RHS
        // in all current Task-2 cases, but we use the general substitution
        // helper for correctness when future rules create nested patterns.
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, measure_id);
        m.set_binding_rhs(bid, new_rhs);
        return Ok(());
    }

    // --- No rule matched: refuse with the op name ---
    let construct = op_name(m, target_node);
    Err(RefuseError {
        node: target_node,
        construct,
        reason: "no determinization rule for this measure-layer construct (Task 3+ needed)"
            .to_string(),
    })
}

/// Match `lawof(draw ?m)` at `node_id`. Returns `Some(m_id)` — the NodeId of
/// the inner measure `?m` — if the pattern matches.
///
/// Shape (confirmed via FlatPIR dump):
///   `lawof` is `Node::Call { head: Builtin("lawof"), args: [draw_node], .. }`
///   `draw`  is `Node::Call { head: Builtin("draw"),  args: [m_node],    .. }`
fn try_beta_law(m: &Module, node_id: NodeId) -> Option<NodeId> {
    let Node::Call(lawof_call) = m.node(node_id) else {
        return None;
    };
    let CallHead::Builtin(lawof_sym) = lawof_call.head else {
        return None;
    };
    if m.resolve(lawof_sym) != "lawof" {
        return None;
    }
    if lawof_call.args.len() != 1 {
        return None;
    }
    let draw_id = lawof_call.args[0];
    let Node::Call(draw_call) = m.node(draw_id) else {
        return None;
    };
    let CallHead::Builtin(draw_sym) = draw_call.head else {
        return None;
    };
    if m.resolve(draw_sym) != "draw" {
        return None;
    }
    if draw_call.args.len() != 1 {
        return None;
    }
    Some(draw_call.args[0])
}

/// Replace all occurrences of `old` with `new_id` in the subtree rooted at
/// `root`. Returns the (possibly new) NodeId for the root after substitution.
fn substitute_in_tree(m: &mut Module, root: NodeId, old: NodeId, new_id: NodeId) -> NodeId {
    map_tree(m, root, &mut |_m, id| (id == old).then_some(new_id))
}

/// Bottom-up leaf-substituting rebuild of the subtree at `root`.
///
/// This is the single shared engine behind [`substitute_in_tree`] (which keys on
/// a target NodeId) and `density::substitute_refs_by_name` (which keys on a
/// `Ref(SelfMod, name)` leaf): both walk the same `children()` enumeration,
/// rebuild only-if-changed, and reconstruct a `Call` from its mapped children.
/// They differ ONLY in the leaf predicate, so that is the injected closure `f`:
/// for each visited node, if `f` returns `Some(replacement)` that node is
/// replaced wholesale (and its children are NOT visited — the caller decides what
/// stands in); if it returns `None`, we recurse into the node's children and
/// rebuild the `Call` when any child changed.
///
/// The arena is append-only: a `Call` whose children are unchanged is reused (no
/// allocation), keeping the arena compact for simple rewrites. Only `Call` nodes
/// carry children (see [`Node::for_each_child`]).
///
/// [`Inputs`](flatppl_core::Inputs) entries are `(Symbol, Ref)` leaves, NOT child
/// sub-nodes, so this walk deliberately does not touch them — it clones them
/// unchanged, exactly as both original functions did. A `Ref` slot in `Inputs`
/// also cannot hold an arbitrary replacement node (it is a name reference, not a
/// `NodeId`), so a leaf mapping that yields a value node has no representable
/// target there; callers that must not silently skip such an input assert on it
/// (see `density::substitute_refs_by_name`).
pub(crate) fn map_tree(
    m: &mut Module,
    root: NodeId,
    f: &mut impl FnMut(&Module, NodeId) -> Option<NodeId>,
) -> NodeId {
    // Leaf replacement decided by the caller's predicate. A replaced node stands
    // in wholesale, so we do NOT descend into its (now-irrelevant) children.
    if let Some(replacement) = f(m, root) {
        return replacement;
    }

    // Collect children and map each; only `Call` nodes have any.
    let children: Vec<NodeId> = m.node(root).children();
    let new_children: Vec<NodeId> = children.iter().map(|&c| map_tree(m, c, f)).collect();

    // If nothing changed, keep the original node.
    if new_children == children {
        return root;
    }

    // Rebuild the Call node with mapped children. Only `Call` nodes have
    // children (see `Node::for_each_child`), so this arm is always reached.
    let Node::Call(orig_call) = m.node(root) else {
        // Non-call nodes have no children — the map would have returned early
        // above (no children, so new_children == children == []).
        unreachable!("non-call node with children is impossible in this IR");
    };
    let head = orig_call.head;
    let inputs = orig_call.inputs.clone();
    let pos_count = orig_call.args.len();
    let named_count = orig_call.named.len();

    // `for_each_child` visits: callee (User head) first, then positional args,
    // then named values — mirror that partition here.
    let (new_head, child_slice) = match head {
        CallHead::User(_) => (CallHead::User(new_children[0]), &new_children[1..]),
        CallHead::Builtin(s) => (CallHead::Builtin(s), &new_children[..]),
    };

    let new_args: Vec<NodeId> = child_slice[..pos_count].to_vec();
    let new_named_values: Vec<NodeId> = child_slice[pos_count..pos_count + named_count].to_vec();

    // Rebuild named args with updated values but original names/kinds.
    let new_named: Vec<flatppl_core::NamedArg> = orig_call
        .named
        .iter()
        .zip(new_named_values.iter())
        .map(|(n, &v)| flatppl_core::NamedArg {
            kind: n.kind,
            name: n.name,
            value: v,
        })
        .collect();

    use flatppl_core::Call;
    m.alloc(Node::Call(Call {
        head: new_head,
        args: new_args.into(),
        named: new_named.into(),
        inputs,
    }))
}

/// True iff `id` is a builtin call whose head is named `op`.
fn is_op(m: &Module, id: NodeId, op: &str) -> bool {
    if let Node::Call(c) = m.node(id) {
        if let CallHead::Builtin(sym) = c.head {
            return m.resolve(sym) == op;
        }
    }
    false
}

/// The op name for a measure-layer node (for refusal messages).
fn op_name(m: &Module, id: NodeId) -> String {
    if let Node::Call(c) = m.node(id) {
        if let CallHead::Builtin(sym) = c.head {
            return m.resolve(sym).to_string();
        }
    }
    // Fallback for Measure/Likelihood-typed non-call nodes.
    format!("{:?}", m.type_of(id))
}

// ---------------------------------------------------------------------------
// Dead-binding sweep — post-logdensityof cleanup
// ---------------------------------------------------------------------------

/// Measure-layer ops that can become dead bindings after `logdensityof`
/// lowering. These are consumed *through* a density rule (the rule inlines a
/// deterministic copy of what it needs) and left behind as orphaned bindings.
///
/// Includes the combinators (`weighted`/`superpose`/… and the `bijection`
/// argument they carry) AND the Kleisli / reification vocabulary a `kchain`
/// marginal consumes: the latent `draw`, the `lawof` of the latent record, the
/// `kernelof` kernel, and the `kchain` node itself. After the marginal is lowered
/// to a self-contained `logsumexp` (with fresh inlined copies of the latent's
/// distribution and the per-atom kernel bodies), none of these are referenced.
const COMBINATOR_OPS: &[&str] = &[
    "weighted",
    "logweighted",
    "superpose",
    "normalize",
    "truncate",
    "pushfwd",
    "bijection",
    // Kleisli-marginal vocabulary (Task 5): orphaned after a kchain density rule.
    "kchain",
    "kernelof",
    "lawof",
    "draw",
    // Independent-product combinator (Task 1 follow-on): `iid(M, N)` is consumed
    // through the density rule (unrolled into per-element terms), so a `d =
    // iid(...)` binding referenced only by the now-lowered `logdensityof` query
    // is orphaned the same way a `weighted`/`superpose` binding is.
    "iid",
    // Positional-joint combinator (Task 2 follow-on): `joint(M1, …, Mk)` is
    // likewise consumed through the density rule (unrolled into per-component
    // terms), so a `d = joint(...)` binding referenced only by the now-lowered
    // `logdensityof` query is orphaned the same way.
    "joint",
    // Likelihood combinator (Task 3 / audit H2): a `obs = likelihoodof(K, data)`
    // binding is unwrapped at the `logdensityof` entry (its `K` is scored at the
    // baked-in `data`), so a binding referenced only by the now-lowered
    // `logdensityof` query is orphaned the same way.
    "likelihoodof",
];

/// After a `logdensityof` rewrite, scan all bindings whose RHS is a measure
/// combinator op.  If no other binding tree references the binding by name,
/// replace its RHS with `0.0` (a harmless Real literal) so the driver loop no
/// longer sees it as a measure-layer node.
///
/// This handles the common pattern:
/// ```text
/// m = weighted(w, Normal(...))
/// a = draw(m)               ← pinned to the scored value after logdensityof lowering
/// lp = logdensityof(...)    ← already lowered
/// ```
/// After lowering, `a` is a Real literal, so `m` is an unreferenced measure
/// binding.  Without this sweep the driver would encounter `weighted` on the
/// next iteration and refuse, since there is no standalone rule for it.
///
/// **Soundness invariant.** A binding is rewritten iff it is BOTH (a) an
/// eliminable measure binding — either a combinator op (`is_combinator_rhs`) OR a
/// `Measure`/`Likelihood`-typed RHS (`is_measure_typed_rhs`) — AND (b) provably
/// unreferenced: no *other* binding subtree contains a `(%ref self name)` to it
/// (`binding_is_referenced`'s BFS).  Zeroing a genuinely-dead measure binding is
/// sound dead-code elimination: it has no observable effect because nothing reads
/// it.  The guard below never touches anything that fails *either* condition, so
/// it cannot disturb a live value, a non-measure binding, or a still-referenced
/// measure.
///
/// The type-based arm generalises past the fixed `COMBINATOR_OPS` op-name list:
/// distribution *constructors* (`Normal`, `Beta`, …) are not a closed set, so a
/// standalone `gauss_x = Normal(mu, sigma)` orphaned after a `likelihoodof`
/// density query (audit H2) cannot be caught by op name.  Keying the second arm
/// on the *inferred type* — which `is_flatpdl` itself uses to reject residual
/// measure-layer values — sweeps exactly the bindings that would otherwise trip
/// the conformance gate, and no others (a value-typed binding never matches).
fn sweep_dead_measure_bindings(m: &mut Module) {
    // Iterate to a fixpoint so dead-binding *cascades* are fully eliminated.
    // A `kchain` marginal leaves a chain `pp = kchain(M, k)` → `k = kernelof(…)`
    // → `z = draw(…)`: only `pp` is unreferenced at first (the still-present-but-
    // dead `k` references `z`, and `pp` references `k`). Zeroing `pp` orphans
    // `k`; zeroing `k` orphans `z`. A `likelihoodof` query leaves an analogous
    // chain `obs = likelihoodof(iid(gauss_x, 1), …)` → `gauss_x = Normal(…)`:
    // sweeping the `obs` combinator orphans the `gauss_x` constructor, caught by
    // the type arm. One pass only kills the currently-orphaned bindings; we
    // repeat until a pass kills nothing.
    loop {
        // Collect bindings whose RHS is eliminable (combinator op OR
        // Measure/Likelihood-typed) AND that no other binding references. Both
        // predicates are read-only over `m`, so we settle the full kill-set
        // before any mutation (a later zeroing in this pass cannot make a
        // previously-live binding look dead).
        let dead: Vec<BindingId> = m
            .bindings()
            .filter(|(bid, b)| {
                is_eliminable_measure_rhs(m, b.rhs) && !binding_is_referenced(m, *bid, b.name)
            })
            .map(|(bid, _)| bid)
            .collect();

        if dead.is_empty() {
            return;
        }

        for bid in dead {
            // Re-assert the invariant at the rewrite site: we only zero a binding
            // that is still an eliminable measure binding and still unreferenced.
            // Cheap, and it documents/enforces that the sweep never rewrites
            // anything else.
            let binding = m.binding(bid);
            debug_assert!(
                is_eliminable_measure_rhs(m, binding.rhs)
                    && !binding_is_referenced(m, bid, binding.name),
                "sweep must only zero a dead measure binding"
            );
            let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
            m.set_binding_rhs(bid, zero);
        }
    }
}

/// True iff `rhs` is an eliminable measure binding: a combinator op
/// (`is_combinator_rhs`) OR a node whose inferred type is `Measure`/`Likelihood`
/// (`is_measure_typed_rhs`). Either alone suffices — the op-name arm catches
/// combinators before inference has classified them, and the type arm catches
/// distribution constructors, which are not on the op-name list.
fn is_eliminable_measure_rhs(m: &Module, rhs: NodeId) -> bool {
    is_combinator_rhs(m, rhs) || is_measure_typed_rhs(m, rhs)
}

/// True iff `rhs`'s inferred type is `Measure` or `Likelihood` — the residual
/// measure-layer value types `is_flatpdl` rejects. Used to sweep an orphaned
/// distribution-constructor binding (`gauss_x = Normal(…)`) whose op name is not
/// on `COMBINATOR_OPS`.
fn is_measure_typed_rhs(m: &Module, rhs: NodeId) -> bool {
    matches!(
        m.type_of(rhs),
        Some(flatppl_core::Type::Measure { .. }) | Some(flatppl_core::Type::Likelihood { .. })
    )
}

/// True iff `rhs` is a builtin call whose head is in [`COMBINATOR_OPS`].
fn is_combinator_rhs(m: &Module, rhs: NodeId) -> bool {
    if let Node::Call(c) = m.node(rhs) {
        if let CallHead::Builtin(sym) = c.head {
            return COMBINATOR_OPS.contains(&m.resolve(sym));
        }
    }
    false
}

/// True iff any binding (other than `bid` itself) contains a `(%ref self name)`
/// node that refers to `name_sym`.
fn binding_is_referenced(m: &Module, bid: BindingId, name_sym: Symbol) -> bool {
    for (other_bid, binding) in m.bindings() {
        if other_bid == bid {
            continue;
        }
        if subtree_contains_ref(m, binding.rhs, name_sym) {
            return true;
        }
    }
    false
}

/// BFS subtree search: returns true iff the subtree at `root` contains a
/// `Ref(SelfMod, name_sym)` node.
fn subtree_contains_ref(m: &Module, root: NodeId, name_sym: Symbol) -> bool {
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            if *name == name_sym {
                return true;
            }
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    false
}
