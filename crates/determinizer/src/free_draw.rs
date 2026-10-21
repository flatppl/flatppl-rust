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

use crate::density::{builtin_name, draw_argument, refuse};
use crate::refuse::RefuseError;
use flatppl_core::{BindingId, Module, Node, NodeId, Ref, RefNs, Symbol};

/// Refuse when a density query reachable from the outputs reads a free draw.
///
/// The outputs are `roots`, or every binding when `roots` is `None`. The walk
/// does not enter reified callables: a query inside one is scored at that
/// callable's inputs, and §04 makes the callable fixed.
pub(crate) fn refuse_free_draws(m: &Module, roots: Option<&[Symbol]>) -> Result<(), RefuseError> {
    let starts: Vec<BindingId> = match roots {
        Some(names) => names.iter().filter_map(|&n| m.binding_by_name(n)).collect(),
        None => m.bindings().map(|(bid, _)| bid).collect(),
    };
    let mut seen = Vec::new();
    for bid in starts {
        visit_binding(m, bid, &mut seen)?;
    }
    Ok(())
}

fn visit_binding(m: &Module, bid: BindingId, seen: &mut Vec<BindingId>) -> Result<(), RefuseError> {
    if seen.contains(&bid) {
        return Ok(());
    }
    seen.push(bid);
    let binding = m.binding(bid);
    visit(m, binding.rhs, binding.name, seen)
}

fn visit(
    m: &Module,
    node: NodeId,
    output: Symbol,
    seen: &mut Vec<BindingId>,
) -> Result<(), RefuseError> {
    match m.node(node) {
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) => match m.binding_by_name(*name) {
            Some(bid) => visit_binding(m, bid, seen),
            None => Ok(()),
        },
        Node::Call(c) => {
            match builtin_name(m, node) {
                Some("lawof" | "functionof" | "kernelof" | "fn") => return Ok(()),
                Some("logdensityof" | "densityof") if !c.args.is_empty() => {
                    if let Some((site, name)) = free_draw(m, c.args[0], &mut Vec::new()) {
                        return Err(refusal(m, site, name, output));
                    }
                }
                _ => {}
            }
            for &arg in c.args.iter() {
                visit(m, arg, output, seen)?;
            }
            for entry in c.named.iter() {
                visit(m, entry.value, output, seen)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// The first draw `measure` reads outside every `lawof`, with the binding that
/// names it when it has one.
fn free_draw(
    m: &Module,
    node: NodeId,
    path: &mut Vec<BindingId>,
) -> Option<(NodeId, Option<Symbol>)> {
    match m.node(node) {
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) => {
            let bid = m.binding_by_name(*name)?;
            if path.contains(&bid) {
                return None;
            }
            let rhs = m.binding(bid).rhs;
            if draw_argument(m, rhs).is_some() {
                return Some((rhs, Some(*name)));
            }
            path.push(bid);
            let found = free_draw(m, rhs, path);
            path.pop();
            found
        }
        Node::Call(c) => {
            if draw_argument(m, node).is_some() {
                return Some((node, None));
            }
            if matches!(
                builtin_name(m, node),
                Some("lawof" | "functionof" | "kernelof" | "fn")
            ) {
                return None;
            }
            c.args
                .iter()
                .chain(c.named.iter().map(|entry| &entry.value))
                .find_map(|&arg| free_draw(m, arg, path))
        }
        _ => None,
    }
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
