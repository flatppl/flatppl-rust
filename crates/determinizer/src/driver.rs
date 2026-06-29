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
use flatppl_core::{BindingId, CallHead, Module, Node, NodeId};

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
    // --- density disintegration: logdensityof(lawof(record(..)), v) → Σ terms ---
    if is_op(m, target_node, "logdensityof") {
        let new_root = crate::density::lower_logdensityof(m, target_node)?;
        let new_rhs = substitute_in_tree(m, m.binding(bid).rhs, target_node, new_root);
        m.set_binding_rhs(bid, new_rhs);
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
///
/// The arena is append-only: we allocate new `Call` nodes whose children have
/// been redirected, walking bottom-up. If no child is `old` the original node
/// is reused (no allocation). This keeps the arena compact for simple rewrites.
fn substitute_in_tree(m: &mut Module, root: NodeId, old: NodeId, new_id: NodeId) -> NodeId {
    if root == old {
        return new_id;
    }

    // Collect children and check if any need rewriting.
    let children: Vec<NodeId> = m.node(root).children();
    let new_children: Vec<NodeId> = children
        .iter()
        .map(|&c| substitute_in_tree(m, c, old, new_id))
        .collect();

    // If nothing changed, keep the original node.
    if new_children == children {
        return root;
    }

    // Rebuild the Call node with substituted children. Only `Call` nodes have
    // children (see `Node::for_each_child`), so this arm is always reached.
    let Node::Call(orig_call) = m.node(root) else {
        // Non-call nodes have no children — substitution would have returned
        // early above (no children, so new_children == children == []).
        unreachable!("non-call node with children is impossible in this IR");
    };
    let head = orig_call.head;
    let inputs = orig_call.inputs.clone();

    // Partition new_children back into positional args and named-arg values.
    // `for_each_child` visits: callee (User head), positional args, named values.
    let (pos_count, named_count) = match head {
        CallHead::User(_) => {
            // first child is the callee node; rest split as (args, named)
            (orig_call.args.len(), orig_call.named.len())
        }
        CallHead::Builtin(_) => (orig_call.args.len(), orig_call.named.len()),
    };

    let (new_head, child_slice) = match head {
        CallHead::User(_) => {
            // children = [callee, args..., named_values...]
            (CallHead::User(new_children[0]), &new_children[1..])
        }
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
