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
use flatppl_infer::ModuleBundle;

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
/// Works on a clone of `m`; the original is not modified. Resolves only
/// same-module refs — a cross-module (`load_module`) measure ref refuses. For a
/// model with loaded dependencies, use [`determinize_with`] and supply the
/// dependency bundle.
pub fn determinize(m: &Module) -> Result<Module, RefuseError> {
    determinize_with(m, &ModuleBundle::new())
}

/// Like [`determinize`], but resolves cross-module (`load_module`) measure refs
/// against `bundle` — the same `ModuleBundle` the host assembled for inference
/// (path → parsed dependency `Module`). A `(%ref <alias> member)` reaching a
/// measure/likelihood-kernel position is grafted from the loaded submodule into
/// the host and then lowered like any local measure (see
/// [`crate::crossmodule`]). With an empty bundle this is byte-identical to
/// [`determinize`] for every self-contained model.
///
/// Works on a clone of `m`; the original is not modified.
pub fn determinize_with(m: &Module, bundle: &ModuleBundle) -> Result<Module, RefuseError> {
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
                apply_rule(&mut work, bid, node_id, bundle)?;
                // Loop: re-scan after the rewrite.
            }
        }
    }
}

/// Scan bindings for the next measure-layer node to reduce. Returns
/// `(binding_id, node_id)` of the chosen node.
///
/// **Two-pass, query-first.** A `logdensityof`/`densityof` query pins the latent
/// `draw`s it scores (it rewrites their bindings to the scored value), so it must
/// fire *before* those bare `draw` bindings are reached and refused: a draw
/// consumed by a density query is reducible *through* that query, not on its own.
/// So we first look for any `logdensityof`/`densityof` node (scanning bindings in
/// source order, outermost-first), and only if neither exists fall back to the
/// general outermost-measure scan (β-law `lawof`, or a refusal target). Without
/// this, the source-order scan would hit a `draw` binding first and refuse before
/// the query that would have legalised it — the exact misleading-refuse-citing-
/// `draw` regression a `densityof` query over a `draw` binding hit before this
/// probe covered it too (only `logdensityof` was probed).
///
/// **`rand` gets the same early-probe treatment**, for the identical reason: a
/// `rand(rng, lawof(record(x = x)))` query samples (and thereby legalises) the
/// `x = draw(...)` binding it closes over, so it must fire before the
/// source-order scan reaches that `draw` binding on its own and refuses it.
fn find_measure_node(m: &Module) -> Option<(BindingId, NodeId)> {
    // `get(disintegrate(…), i)` gets the highest priority (see
    // `find_get_disintegrate`): the `get` must be eliminated into the structural
    // split *before* a `logdensityof`/`bayesupdate` consumes it (and hits the
    // primitive-constructor refuse in `lower_measure_density`) or the general scan
    // reaches the bare `disintegrate` and refuses it.
    if let Some(hit) = find_get_disintegrate(m) {
        return Some(hit);
    }
    // `restrict(M, x)` is desugared (§06 "Measure restriction") into
    // `bayesupdate(likelihoodof(kernel, x), marginal)` over the disintegration on
    // `x`'s field names. Like `get(disintegrate, i)`, it must fire BEFORE a
    // `logdensityof` consumes it: a `logdensityof(restrict(…), θ)` would otherwise
    // reach the primitive-constructor refuse in `lower_measure_density` (which
    // keeps a `restrict` safety-net arm). Eliminating the `restrict` first leaves
    // a `bayesupdate` the density query lowers via the existing posterior path.
    if let Some(hit) = find_op_node(m, &["restrict"]) {
        return Some(hit);
    }
    // `logdensityof` and `densityof` share the same query-first priority: a
    // `densityof(M, v)` pins latent draws exactly like `logdensityof(M, v)` does
    // (`densityof` lowers via the same core dispatch, wrapped in `exp`), so both
    // op names must be probed here before the general scan reaches a `draw`.
    if let Some(hit) = find_op_node(m, &["logdensityof", "densityof"]) {
        return Some(hit);
    }
    if let Some(hit) = find_op_node(m, &["rand"]) {
        return Some(hit);
    }
    for (bid, binding) in m.bindings() {
        if let Some(id) = find_in_subtree(m, binding.rhs) {
            return Some((bid, id));
        }
    }
    None
}

/// Find the first node (outermost, BFS) whose builtin head is named one of
/// `ops`, scanning bindings in source order.
fn find_op_node(m: &Module, ops: &[&str]) -> Option<(BindingId, NodeId)> {
    for (bid, binding) in m.bindings() {
        let mut queue = vec![binding.rhs];
        let mut qi = 0;
        while qi < queue.len() {
            let id = queue[qi];
            qi += 1;
            if let Node::Call(c) = m.node(id) {
                if let CallHead::Builtin(sym) = c.head {
                    if ops.contains(&m.resolve(sym)) {
                        return Some((bid, id));
                    }
                }
            }
            m.for_each_child(id, |child| queue.push(child));
        }
    }
    None
}

/// Scan bindings (source order) for the first `get(disintegrate(…), i)` node —
/// the highest-priority driver target. Eliminating this `get` into the structural
/// disintegration split (`split_disintegrate`) must precede the `logdensityof`
/// scan: a `disintegrate` tuple is consumed by `forward_kernel = get(D, 1)` /
/// `prior = get(D, 2)`, and if the density query lowered first it would reach the
/// still-present `get` through the primitive-constructor path and refuse.
fn find_get_disintegrate(m: &Module) -> Option<(BindingId, NodeId)> {
    for (bid, binding) in m.bindings() {
        let mut queue = vec![binding.rhs];
        let mut qi = 0;
        while qi < queue.len() {
            let id = queue[qi];
            qi += 1;
            if match_get_disintegrate(m, id).is_some() {
                return Some((bid, id));
            }
            m.for_each_child(id, |c| queue.push(c));
        }
    }
    None
}

/// Match `get(D, i)` where `D` resolves (one `(%ref self …)` hop) to a
/// `disintegrate(…)` call. Returns `(disintegrate_node, i)` — the resolved
/// disintegrate call node and the 1-based tuple index literal — or `None` when
/// the node is not a well-formed `get` over a disintegrate.
fn match_get_disintegrate(m: &Module, node: NodeId) -> Option<(NodeId, i64)> {
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "get" || c.args.len() != 2 || !c.named.is_empty() {
        return None;
    }
    // arg0 must resolve (one hop) to a `disintegrate(…)` call.
    let (target, _) = crate::density::resolve_ref_one(m, c.args[0]);
    if crate::density::builtin_name(m, target) != Some("disintegrate") {
        return None;
    }
    // arg1 is the 1-based tuple index literal (`get(tmp, k+1)`, syntax/parser.rs).
    let Node::Lit(Scalar::Int(i)) = m.node(c.args[1]) else {
        return None;
    };
    Some((target, *i))
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
/// The cross-module-graft branch (`graft_query_target`) is the exception — it
/// can add many nodes and leaves the `logdensityof` in place for the next
/// iteration — and instead relies on the progress invariant documented at
/// `graft_query_target`: the query target goes from a module-ref to a local
/// node, so a given query can graft at most once.
fn apply_rule(
    m: &mut Module,
    bid: BindingId,
    target_node: NodeId,
    bundle: &ModuleBundle,
) -> Result<(), RefuseError> {
    // --- structural disintegration: get(disintegrate(sel, lawof(record …)), i) ---
    // The pinned bi3 IR consumes a `disintegrate` tuple through two `get`s —
    // `forward_kernel = get(D, 1)` (the KERNEL, tuple elem 1) and
    // `prior = get(D, 2)` (the MARGINAL, tuple elem 2), 1-based per the tuple-
    // destructuring desugar (`get(tmp, k+1)`, syntax/parser.rs:437). Replace the
    // `get` with the matching component of the structural split so the downstream
    // `likelihoodof(kernel, obs)` / `bayesupdate(L, marginal)` lower via the
    // existing paths (spec §06 "Structural disintegration"). This arm fires before
    // the `logdensityof` arm because `find_measure_node` returns the `get` first.
    if let Some((disint_node, index)) = match_get_disintegrate(m, target_node) {
        // Only the 1-based kernel (1) / marginal (2) projections of the pair are
        // meaningful; any other index is out of range — refuse, don't mislower.
        if index != 1 && index != 2 {
            return Err(RefuseError {
                node: target_node,
                construct: "get".to_string(),
                reason: format!(
                    "get index {index} out of range for a disintegrate (kernel, marginal) tuple"
                ),
            });
        }
        // Structural split; a non-explicit-DAG disintegrate (§06 permits refusing
        // intractable / non-`lawof(record)` disintegrations) yields None → refuse.
        let (kernel, marginal) = crate::disintegrate::split_disintegrate(m, disint_node)
            .ok_or_else(|| RefuseError {
                node: disint_node,
                construct: "disintegrate".to_string(),
                reason: "disintegrate is not an explicit `lawof(record(…))` structural split"
                    .to_string(),
            })?;
        let replacement = if index == 1 { kernel } else { marginal };
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, replacement);
        m.set_binding_rhs(bid, new_rhs);
        // Once BOTH gets are eliminated the `D = disintegrate(…)` binding (and the
        // `joint_model = lawof(record(…))` it consumes) are unreferenced. The split
        // components carry verbatim `(%ref self …)` value nodes to the draws, so the
        // draw bindings stay referenced and survive; only the disintegrate/joint
        // scaffold is dead. Sweep it (disintegrate is on `COMBINATOR_OPS`) so the
        // general scan never reaches the bare `disintegrate` and refuses — this is
        // what makes each rewrite strictly remove a `get`/`disintegrate` node and
        // the driver terminate.
        sweep_dead_measure_bindings(m);
        return Ok(());
    }

    // --- measure restriction: restrict(M, x) → bayesupdate(likelihoodof(kernel, x), marginal) ---
    // Spec §06 "Measure restriction": the non-normalized conditional of `M` given
    // the observed values `x` desugars — via structural disintegration on `x`'s
    // field names — into `bayesupdate(likelihoodof(kernel, x), marginal)` where
    // `(kernel, marginal) = disintegrate([field-names of x], M)`. This arm fires
    // before the `logdensityof` arm (`find_measure_node` returns the `restrict`
    // first), so the downstream `bayesupdate` lowers via the existing posterior
    // path. A shape the desugaring cannot handle (`x` not a record, a field of `x`
    // naming no variate of `M`, or a non-`lawof(record)` `M`) yields `None` →
    // refuse, don't mislower.
    if is_op(m, target_node, "restrict") {
        let desugared =
            crate::disintegrate::rewrite_restrict(m, target_node).ok_or_else(|| RefuseError {
                node: target_node,
                construct: "restrict".to_string(),
                reason: "restrict(M, x) is not the explicit structural case: x must be a \
                         record of observed values whose fields are all variates of a \
                         lawof(record(…)) M"
                    .to_string(),
            })?;
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, desugared);
        m.set_binding_rhs(bid, new_rhs);
        // The desugared `bayesupdate` carries the split kernel/marginal, whose
        // verbatim `(%ref self …)` value nodes reach the draws directly, so the
        // `M = lawof(record(…))` joint binding is now unreferenced (dead
        // scaffold). Sweep it (`lawof` is on `COMBINATOR_OPS`) so the general scan
        // never reaches the bare `lawof(record(…))` and refuses.
        sweep_dead_measure_bindings(m);
        return Ok(());
    }

    // --- density disintegration: logdensityof(lawof(M), v) / densityof(lawof(M), v) → deterministic density ---
    // `densityof` shares this whole arm with `logdensityof` (it lowers via the
    // same [`crate::density::lower_density_core`] dispatch, wrapped in `exp` —
    // §06: `densityof(M,x) = exp(logdensityof(M,x))`); only the final lowering
    // call differs.
    let is_logdensityof = is_op(m, target_node, "logdensityof");
    let is_densityof = is_op(m, target_node, "densityof");
    if is_logdensityof || is_densityof {
        // Defer-and-reloop for a cross-module query TARGET (GAP D). When the
        // query's measure target is (or resolves via one `(%ref self …)` hop to)
        // a cross-module `(%ref <alias> member)`, graft the referenced submodule
        // subtree into the host and rewrite the query to point at the LOCAL
        // grafted node — WITHOUT lowering yet. Returning here lets the driver loop
        // re-run inference at the top of the NEXT iteration, which types the
        // freshly-grafted subtree (crucially an `iid`'s const-evaluated domain
        // shape). The re-scan then finds this same query with a now-local, typed
        // target, `graft_query_target` returns `None`, and it lowers fully with a
        // resolvable `iid_static_size`; the post-lowering sweep likewise runs
        // after the grafted (now-typed) dead kernel binding has been classified,
        // so it is swept rather than surviving into the conformance check. Without
        // this deferral the old inline graft-then-lower ran on still-untyped nodes
        // in a single call (iid-size refuse / KernelNotBuiltinArg refuse).
        // `graft_query_target` rebuilds the deferred query with the SAME op name
        // (`logdensityof` or `densityof`) it was given, so this branch is shared.
        if let Some(new_query) = crate::density::graft_query_target(m, target_node, bundle)? {
            let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, new_query);
            m.set_binding_rhs(bid, new_rhs);
            // The intermediate `x = m.L` self-ref binding (if any) and the
            // `helpers = load_module(…)` binding may now be dead; sweep the
            // measure-typed ones so the next scan is clean.
            sweep_dead_measure_bindings(m);
            return Ok(());
        }
        let new_root = if is_logdensityof {
            crate::density::lower_logdensityof(m, target_node, bundle)?
        } else {
            crate::density::lower_densityof(m, target_node, bundle)?
        };
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, new_root);
        m.set_binding_rhs(bid, new_rhs);
        // After lowering a logdensityof/densityof query, some measure bindings
        // (e.g. `m = weighted(...)`) may now be dead code — the draw was pinned
        // and the combinator binding is no longer referenced.  Sweep them out now
        // so the outer scan loop does not encounter them as unhandled
        // measure-layer nodes on the next iteration.
        sweep_dead_measure_bindings(m);
        return Ok(());
    }

    // --- rand / builtin_sample: sampling-side slice ---
    // `rand(rng, lawof(M))` threads an RNG through M's generative subgraph,
    // replacing each `draw(mᵢ)` with `builtin_sample(rngᵢ, mᵢ, inputᵢ)` (spec §07).
    if is_op(m, target_node, "rand") {
        let new_root = crate::sample::lower_rand(m, bid, target_node)?;
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, new_root);
        m.set_binding_rhs(bid, new_rhs);
        // As with `logdensityof`, sampling a draw leaves its `x = draw(...)`
        // binding referenced by nothing (the sampled value is a fresh inline
        // node, not a ref to `x`) — sweep it out before the next scan.
        sweep_dead_measure_bindings(m);
        return Ok(());
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
///
/// Cross-reference: `density.rs::MEASURE_COMBINATOR_OPS` encodes the same
/// measure-combinator vocabulary for a DIFFERENT purpose (rejecting a composed
/// truncation base in the `normalize(truncate)` CDF-Z path) and intentionally
/// has different membership — this list is about DCE eligibility, not leaf-vs-
/// combinator classification. A new measure-algebra op may need adding to
/// both lists — check `density.rs` too.
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
    // Likelihood combinator (Task 3 / measure-algebra-audit.md H2): a
    // `obs = likelihoodof(K, data)`
    // binding is unwrapped at the `logdensityof` entry (its `K` is scored at the
    // baked-in `data`), so a binding referenced only by the now-lowered
    // `logdensityof` query is orphaned the same way.
    "likelihoodof",
    // Structural-disintegration scaffold (Task 4): after `get(D, 1)` / `get(D, 2)`
    // are eliminated into the split kernel/marginal, the `D = disintegrate(…)`
    // binding is unreferenced. It types to a `%tuple`, not a `Measure`, so the
    // type arm below does not catch it — the op-name arm sweeps the dead scaffold
    // (its `lawof(record(…))` joint argument is caught by the existing `lawof`
    // entry once `D` is gone). Only zeroed when unreferenced, so a `disintegrate`
    // whose `get`s are still present is never swept.
    "disintegrate",
    // Likelihood-combining op (§06 "Combining likelihoods"): a
    // `L = joint_likelihood(L1, …, Lk)` binding is unwrapped at the
    // `logdensityof` entry (each component scored at the shared θ and summed), so
    // it is orphaned the same way. It is REQUIRED on the op-name list — unlike a
    // `likelihoodof`, a `joint_likelihood` may infer to `%deferred` (its obstype
    // resolution is a separate infer limitation), so the `Measure`/`Likelihood`
    // type arm would not catch it; without this, the dead `L` binding survives
    // and keeps its components referenced, so the driver later trips over a
    // standalone component `likelihoodof`.
    "joint_likelihood",
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
/// density query (measure-algebra-audit.md H2) cannot be caught by op name.
/// Keying the second arm
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

/// True iff `rhs`'s inferred type is `Measure`, `Likelihood`, or `Kernel` — the
/// residual measure-layer value types `is_flatpdl` rejects (it flags a `Kernel`
/// outside a `builtin_*` argument, so a bare `Kernel`-typed binding RHS is
/// rejected exactly as a `Measure`/`Likelihood` one is). Used to sweep an orphaned
/// distribution-constructor binding (`gauss_x = Normal(…)`) whose op name is not on
/// `COMBINATOR_OPS`, and a dead reified-measure binding
/// (`k = functionof(broadcast(Poisson, …), …)`, `Kernel`-typed) left behind once a
/// likelihood query over it is lowered: `functionof` is deliberately NOT on
/// `COMBINATOR_OPS` / `MEASURE_VOCAB` (a `functionof` over a deterministic body is
/// a legal FlatPDL FUNCTION, `Function`-typed — never matched here), so the
/// reified-MEASURE case (always `Kernel`-typed) is caught by this type arm instead.
fn is_measure_typed_rhs(m: &Module, rhs: NodeId) -> bool {
    matches!(
        m.type_of(rhs),
        Some(flatppl_core::Type::Measure { .. })
            | Some(flatppl_core::Type::Likelihood { .. })
            | Some(flatppl_core::Type::Kernel { .. })
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
/// `Ref(SelfMod, name_sym)` node — as a body sub-node OR as a `functionof` /
/// `kernelof` reification *input* boundary entry.
///
/// **`Inputs`-aware (sweep-only).** `for_each_child` / `children()` deliberately
/// EXCLUDE a `Call`'s [`flatppl_core::Inputs`] bucket (core `node.rs`), so a
/// constructor/kernel binding referenced ONLY through a `(name, %ref self <name>)`
/// reification input — e.g. `k = kernelof(pushfwd(f, g), …)` closing over `g =
/// Normal(…)` — would look UNREFERENCED to a body-only walk, and the dead-binding
/// sweep would zero it, leaving the live reification closing over a zeroed
/// constructor. So this walk additionally scans each `Call`'s `inputs` entries.
/// It only makes "referenced" MORE inclusive, so the sweep zeroes strictly fewer
/// bindings — a sound tightening. This `Inputs`-scanning behaviour is scoped to
/// the sweep, which is `subtree_contains_ref`'s only caller
/// ([`binding_is_referenced`]).
fn subtree_contains_ref(m: &Module, root: NodeId, name_sym: Symbol) -> bool {
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        match m.node(id) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) if *name == name_sym => return true,
            Node::Call(c) => {
                // A reification input `(name, %ref self <name>)` references the
                // binding just as a body ref does — but lives outside `children()`.
                if let Some(flatppl_core::Inputs::Spec(entries)) = &c.inputs {
                    for (_, r) in entries.iter() {
                        if r.ns == RefNs::SelfMod && r.name == name_sym {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatppl_core::{Binding, Call, Inputs, Mass, NamedArg, Type};

    /// A `Measure`-typed constructor binding referenced ONLY through another
    /// binding's `functionof` / `kernelof` reification `Inputs` boundary entry
    /// (`(g, %ref self g)`) must NOT be swept as dead. `children()` excludes the
    /// `Inputs` bucket (core `node.rs`), so a body-only reference check judged `g`
    /// unreferenced and the type arm zeroed it — leaving the live reification `k`
    /// closing over a zeroed constructor. The `Inputs`-aware `subtree_contains_ref`
    /// sees the boundary reference, so the sweep leaves `g` alone. This drives the
    /// sweep directly (the full `determinize` over such a shape refuses anyway,
    /// because the surviving `g` is a residual measure-layer binding — the point
    /// here is that the sweep does not UNSOUNDLY zero it out from under `k`).
    #[test]
    fn sweep_preserves_binding_referenced_only_via_reification_input() {
        let mut m = Module::new();

        // g = Normal(mu=0.0, sigma=1.0) — a Measure-typed constructor.
        let normal = m.intern("Normal");
        let mu = m.intern("mu");
        let sigma = m.intern("sigma");
        let z0 = m.alloc(Node::Lit(Scalar::Real(0.0)));
        let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
        let g_rhs = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(normal),
            args: Vec::<NodeId>::new().into(),
            named: vec![
                NamedArg {
                    kind: flatppl_core::NamedKind::Kwarg,
                    name: mu,
                    value: z0,
                },
                NamedArg {
                    kind: flatppl_core::NamedKind::Kwarg,
                    name: sigma,
                    value: one,
                },
            ]
            .into(),
            inputs: None,
        }));
        m.set_type(
            g_rhs,
            Type::Measure {
                domain: Box::new(Type::Scalar(flatppl_core::ScalarType::Real)),
                mass: Mass::Normalized,
            },
        );
        let g_name = m.intern("g");
        let g_bid = m.add_binding(Binding {
            name: g_name,
            rhs: g_rhs,
            doc: None,
            public: true,
            synthetic: false,
        });

        // k = functionof(_x_, x = _x_, g = g) — references `g` ONLY via its
        // reification `Inputs`, never in the body.
        let functionof = m.intern("functionof");
        let x = m.intern("x");
        let ph_x = m.intern("_x_");
        let body = m.alloc(Node::Ref(Ref {
            ns: RefNs::Local,
            name: ph_x,
        }));
        let k_rhs = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(functionof),
            args: vec![body].into(),
            named: Vec::<NamedArg>::new().into(),
            inputs: Some(Inputs::Spec(
                vec![
                    (
                        x,
                        Ref {
                            ns: RefNs::Local,
                            name: ph_x,
                        },
                    ),
                    (
                        g_name,
                        Ref {
                            ns: RefNs::SelfMod,
                            name: g_name,
                        },
                    ),
                ]
                .into(),
            )),
        }));
        let k_name = m.intern("k");
        let _k_bid = m.add_binding(Binding {
            name: k_name,
            rhs: k_rhs,
            doc: None,
            public: true,
            synthetic: false,
        });

        sweep_dead_measure_bindings(&mut m);

        // `g` must survive: still a `Normal` call, NOT the `0.0` sweep sentinel.
        let g_after = m.binding(g_bid).rhs;
        assert!(
            matches!(m.node(g_after), Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if m.resolve(s) == "Normal")),
            "g referenced only via k's reification Inputs must not be swept: got {:?}",
            m.node(g_after)
        );
    }
}
