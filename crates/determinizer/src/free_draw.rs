//! The free-draw check on density outputs.
//!
//! §13 "Output reduction": "`draw` nodes take their values from the explicit
//! `point`, unless marginalized out". A density query whose measure reads a draw
//! outside every `lawof` takes it from neither, so the output has no value. The
//! engine applies no implicit marginal and reads no other output's point.
//!
//! The check runs on the source module, before any query pins a latent. A pin
//! rewrites the shared binding, so a later check could not tell a draw from the
//! point another output scored it at.

use std::collections::HashSet;

use crate::density::{builtin_name, draw_argument, refuse};
use crate::refuse::RefuseError;
use flatppl_core::{BindingId, Module, Node, NodeId, Ref, RefNs, Symbol};

/// Refuse when a density query reachable from the outputs reads a free draw.
///
/// The outputs are `roots`, or every binding when `roots` is `None`. The walk
/// does not enter reified callables: a query inside one is scored at that
/// callable's inputs, and §04 makes the callable fixed.
///
/// Both walks are iterative and visit each binding once, so a deep chain of
/// bindings cannot overflow the stack and shared bindings are not re-expanded.
/// Children are pushed in reverse, so the traversal and hence the reported
/// draw match a recursive pre-order walk.
pub(crate) fn refuse_free_draws(m: &Module, roots: Option<&[Symbol]>) -> Result<(), RefuseError> {
    let starts: Vec<BindingId> = match roots {
        Some(names) => names.iter().filter_map(|&n| m.binding_by_name(n)).collect(),
        None => m.bindings().map(|(bid, _)| bid).collect(),
    };
    let mut seen = HashSet::new();
    // Bindings a finished query walk proved free of draws. The walk is
    // context-free, so a later query can skip them.
    let mut clean = HashSet::new();
    let mut stack = Vec::new();
    for bid in starts {
        stack.push(Item::Binding(bid));
        while let Some(item) = stack.pop() {
            let (node, output) = match item {
                Item::Binding(bid) => {
                    if !seen.insert(bid) {
                        continue;
                    }
                    let binding = m.binding(bid);
                    (binding.rhs, binding.name)
                }
                Item::Node(node, output) => (node, output),
            };
            match m.node(node) {
                Node::Ref(Ref {
                    ns: RefNs::SelfMod,
                    name,
                }) => {
                    if let Some(bid) = m.binding_by_name(*name) {
                        stack.push(Item::Binding(bid));
                    }
                }
                Node::Call(c) => {
                    match builtin_name(m, node) {
                        Some("lawof" | "functionof" | "kernelof" | "fn") => continue,
                        Some("logdensityof" | "densityof") if !c.args.is_empty() => {
                            if let Some((site, name)) = free_draw(m, c.args[0], &mut clean) {
                                return Err(refusal(m, site, name, output));
                            }
                        }
                        _ => {}
                    }
                    let children = c
                        .args
                        .iter()
                        .chain(c.named.iter().map(|entry| &entry.value));
                    let first = stack.len();
                    stack.extend(children.map(|&arg| Item::Node(arg, output)));
                    stack[first..].reverse();
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// A pending node of the output walk, or a binding whose right-hand side it
/// reaches under that binding's own name.
enum Item {
    Binding(BindingId),
    Node(NodeId, Symbol),
}

/// The first draw `measure` reads outside every `lawof`, in pre-order, with the
/// binding that names it when it has one.
///
/// `visited` per call doubles as the cycle guard. A binding reached a second
/// time already yielded no draw, or the walk would have returned. When the
/// walk finds no draw, every binding it visited joins `clean`.
fn free_draw(
    m: &Module,
    measure: NodeId,
    clean: &mut HashSet<BindingId>,
) -> Option<(NodeId, Option<Symbol>)> {
    let mut visited = HashSet::new();
    let mut stack = vec![measure];
    while let Some(node) = stack.pop() {
        match m.node(node) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => {
                let Some(bid) = m.binding_by_name(*name) else {
                    continue;
                };
                if clean.contains(&bid) || !visited.insert(bid) {
                    continue;
                }
                let rhs = m.binding(bid).rhs;
                if draw_argument(m, rhs).is_some() {
                    return Some((rhs, Some(*name)));
                }
                stack.push(rhs);
            }
            Node::Call(c) => {
                if draw_argument(m, node).is_some() {
                    return Some((node, None));
                }
                if matches!(
                    builtin_name(m, node),
                    Some("lawof" | "functionof" | "kernelof" | "fn")
                ) {
                    continue;
                }
                let first = stack.len();
                stack.extend(
                    c.args
                        .iter()
                        .chain(c.named.iter().map(|entry| &entry.value)),
                );
                stack[first..].reverse();
            }
            _ => {}
        }
    }
    clean.extend(visited);
    None
}

fn refusal(m: &Module, site: NodeId, name: Option<Symbol>, output: Symbol) -> RefuseError {
    let output = m.resolve(output);
    let reason = match name {
        Some(name) => {
            let name = m.resolve(name);
            format!(
                "the density query in `{output}` reads draw `{name}`, which is neither in its \
                 point nor integrated out by a `lawof` (§13 \"Output reduction\"); put `{name}` \
                 in the point, or score `lawof(...)` of the value instead"
            )
        }
        None => format!(
            "the density query in `{output}` reads an inline `draw`, which is neither in its \
             point nor integrated out by a `lawof` (§13 \"Output reduction\"); bind it, then \
             put it in the point or score `lawof(...)` of the value instead"
        ),
    };
    refuse(site, m, &reason)
}
