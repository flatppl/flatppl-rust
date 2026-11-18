//! Stochastic ancestors of measures in a typed FlatPPL graph.

use flatppl_core::graph::draw_argument;
use flatppl_core::{BindingId, CallHead, Module, Node, NodeId, Ref, RefNs, Symbol};

/// A draw reached while tracing the parameters of a measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StochasticAncestor {
    /// The draw site in the original declaration. Identically parameterized draws can be distinct.
    pub draw: NodeId,
    /// The binding that declares this draw, if it has one.
    pub name: Option<Symbol>,
    pub prior: NodeId,
}

/// Trace measure operands to their first draws, in argument order.
/// Repeated references retain their multiplicity. An observed draw site or
/// conditioned input ends the walk. A reified law absorbs its internal draws.
/// Earlier query pins use the binding's original declaration.
pub fn measure_stochastic_ancestors(
    m: &Module,
    node: NodeId,
    observed: &[NodeId],
) -> Vec<StochasticAncestor> {
    // `via` is the binding symbol of the ref hop taken IMMEDIATELY above `node`, and
    // only that one: a draw reached through an intervening call (`mu = 2 * z`) is
    // attributed to its own ref hop, never to the outer binding.
    fn walk(
        m: &Module,
        node: NodeId,
        via: Option<Symbol>,
        observed: &[NodeId],
        path: &mut Vec<BindingId>,
        out: &mut Vec<StochasticAncestor>,
    ) {
        match m.node(node) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => {
                // An input the enclosing reification declares is a GIVEN, not a
                // stochastic ancestor: §04 "Reification to measures" makes it the
                // callable's argument, and §06 `likelihoodof` fixes
                // `densityof(likelihoodof(K, obs), theta)` as `pdf(K(theta), obs)` —
                // conditional on θ. Marginalizing it would score the obs against the
                // model's prior predictive instead.
                if m.is_conditioned_input(*name) {
                    return;
                }
                let Some(bid) = m.binding_by_name(*name) else {
                    return;
                };
                if path.contains(&bid) {
                    return; // cyclic definition — stop rather than recurse forever
                }
                path.push(bid);
                // `declared_binding_rhs`, so a latent an earlier query pinned walks as
                // the `draw(prior)` the pin overwrote. One arm for both, rather than a
                // pinned special case: the special case pushed its ancestor without
                // consulting `observed`, so a SIBLING field's latent counted as an
                // outside ancestor once pinned, and the record law refused.
                walk(
                    m,
                    m.declared_binding_rhs(bid),
                    Some(*name),
                    observed,
                    path,
                    out,
                );
                path.pop();
            }
            Node::Call(c) => {
                // A `draw` IS the stochastic ancestor; no need to look inside it.
                if let Some(prior) = draw_argument(m, node) {
                    if !observed.contains(&node) {
                        out.push(StochasticAncestor {
                            draw: node,
                            name: via,
                            prior,
                        });
                    }
                    return;
                }
                if matches!(c.head, CallHead::Builtin(s) if m.resolve(s) == "lawof") {
                    return;
                }
                for &arg in c.args.iter() {
                    walk(m, arg, None, observed, path, out);
                }
                for n in c.named.iter() {
                    walk(m, n.value, None, observed, path, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(m, node, None, observed, &mut Vec::new(), &mut out);
    out
}

/// Return the common binding and prior when every occurrence is the same named draw.
pub fn sole_named_ancestor(ancestors: &[StochasticAncestor]) -> Option<(Symbol, NodeId)> {
    let (first, rest) = ancestors.split_first()?;
    let name = first.name?;
    rest.iter()
        .all(|other| other == first)
        .then_some((name, first.prior))
}
