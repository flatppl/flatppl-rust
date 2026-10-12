use std::collections::{HashMap, HashSet};

use flatppl_core::{CallHead, Mass, Node, NodeId, Phase, RefNs, Symbol, Type};

use crate::modules::Resolved;
use crate::rule::RuleStep;
use crate::trace::Inferencer;

/// Recompute closed laws after an improper ancestor becomes a kernel input.
pub(super) fn conditional_reification(
    inf: &Inferencer<'_, '_>,
    node: NodeId,
    result: (Type, Phase),
) -> RuleStep<(Type, Phase)> {
    if matches!(result.0, Type::Failed(_)) {
        return RuleStep::Ready(result);
    }
    let mut seeds = Vec::new();
    let mut improper_input = false;
    for name in super::reification_boundary_targets(inf, node) {
        let Some(binding) = inf.module.binding_by_name(name) else {
            continue;
        };
        let target = inf.module.binding(binding).rhs;
        if inf.is_substituted(target) {
            continue;
        }
        let Some(ty) = inf.lookup_type(target) else {
            continue;
        };
        improper_input |= !matches!(
            law_mass(inf, target, &[]),
            Mass::Normalized | Mass::Deferred
        );
        seeds.push((
            target,
            Resolved {
                ty: ty.clone(),
                phase: Phase::Parameterized,
                vset: inf.lookup_valueset(target),
                result: None,
                catalogue: None,
                admitted_source: admitted(inf, target),
            },
        ));
    }
    if !improper_input {
        return RuleStep::Ready(result);
    }
    // Existing mass rules now see the boundary before tracing nested laws.
    // The substituted targets prevent this request from repeating in that scope.
    RuleStep::InferBody {
        body: node,
        seeds,
        targets: None,
        resume: Box::new(move |_, (ty, _)| RuleStep::Ready((ty, result.1))),
    }
}

/// Admission follows the source construction, never an improper mass label.
pub(crate) fn admitted(inf: &Inferencer<'_, '_>, node: NodeId) -> bool {
    source(inf, node, &mut HashMap::new(), 0)
}

fn source(
    inf: &Inferencer<'_, '_>,
    node: NodeId,
    memo: &mut HashMap<NodeId, bool>,
    depth: u32,
) -> bool {
    if depth > 64 {
        return false;
    }
    if let Some(&admitted) = memo.get(&node) {
        return admitted;
    }
    if mass_of(inf, node) == Some(Mass::Normalized) {
        return true;
    }
    if let Some(admitted) = inf.source_admission(node) {
        return admitted;
    }
    if inf.is_substituted(node) {
        return false;
    }
    // An in-progress source cannot prove admission through a cycle.
    memo.insert(node, false);
    let admitted = match inf.module.node(node) {
        Node::Ref(r) if r.ns == RefNs::SelfMod => inf
            .module
            .binding_by_name(r.name)
            .is_some_and(|b| source(inf, inf.module.binding(b).rhs, memo, depth + 1)),
        Node::Call(call) => match call.head {
            CallHead::User(callee) => {
                let callee = super::resolve_binding_refs(inf, callee);
                if let Some(reference) = inf.module_catalogue_ref(callee) {
                    reference.module == "regression" && reference.member == "Flat"
                } else if inf.source_admission(callee) == Some(true) {
                    true
                } else if let Some((reified, _)) = super::local_reification(inf, callee) {
                    source(inf, reified, memo, depth + 1)
                } else {
                    false
                }
            }
            CallHead::Builtin(head) => match inf.module.resolve(head) {
                "iid" => call
                    .args
                    .first()
                    .copied()
                    .or_else(|| {
                        call.named
                            .iter()
                            .find(|arg| inf.module.resolve(arg.name) == "M")
                            .map(|arg| arg.value)
                    })
                    .is_some_and(|base| source(inf, base, memo, depth + 1)),
                "joint" | "jointchain" => {
                    (!call.args.is_empty() || !call.named.is_empty())
                        && call
                            .args
                            .iter()
                            .copied()
                            .chain(call.named.iter().map(|arg| arg.value))
                            .all(|part| source(inf, part, memo, depth + 1))
                }
                "functionof" => call
                    .args
                    .first()
                    .is_some_and(|&body| source(inf, body, memo, depth + 1)),
                "lawof" | "kernelof" => call.args.first().is_some_and(|&body| {
                    let boundary = super::reification_boundary_targets(inf, node);
                    law_source(inf, body, &boundary, memo, depth + 1)
                }),
                _ => false,
            },
        },
        _ => false,
    };
    memo.insert(node, admitted);
    admitted
}

pub(crate) fn admitted_value_law(inf: &Inferencer<'_, '_>, node: NodeId) -> bool {
    law_source(inf, node, &[], &mut HashMap::new(), 0)
}

fn law_source(
    inf: &Inferencer<'_, '_>,
    body: NodeId,
    boundary: &[Symbol],
    memo: &mut HashMap<NodeId, bool>,
    depth: u32,
) -> bool {
    mass_of(inf, body).is_none_or(|_| source(inf, body, memo, depth + 1))
        && draw_sources(inf, body, boundary)
            .iter()
            .all(|&measure| source(inf, measure, memo, depth + 1))
}

fn mass_of(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<Mass> {
    match inf.lookup_type(node)? {
        Type::Measure { mass, .. } | Type::Kernel { mass, .. } => Some(*mass),
        _ => None,
    }
}

/// A conditional measure's own mass and every unbound draw source enter its law.
pub(super) fn law_mass(inf: &Inferencer<'_, '_>, node: NodeId, boundary: &[Symbol]) -> Mass {
    let sources = draw_sources(inf, node, boundary);
    let ancestors = source_mass(inf, &sources);
    if let Some(mass) = mass_of(inf, node) {
        return if ancestors == Mass::Normalized {
            mass
        } else {
            combine_mass(mass, ancestors)
        };
    }
    // The identity law can retain local finiteness. An arbitrary deterministic
    // map cannot: mapping Flat to zero produces an infinite Dirac measure.
    let mut resolved = node;
    let mut visited = HashSet::new();
    while visited.insert(resolved) && !inf.is_substituted(resolved) {
        match inf.module.node(resolved) {
            Node::Ref(r) if r.ns == RefNs::SelfMod && !boundary.contains(&r.name) => {
                let Some(binding) = inf.module.binding_by_name(r.name) else {
                    break;
                };
                resolved = inf.module.binding(binding).rhs;
            }
            Node::Call(call) if matches!(call.head, CallHead::Builtin(head) if inf.module.resolve(head) == "draw") =>
            {
                if let Some(&measure) = call.args.first()
                    && matches!(
                        inf.lookup_phase(measure),
                        Some(Phase::Fixed | Phase::Parameterized)
                    )
                {
                    return mass_of(inf, measure).unwrap_or(Mass::Deferred);
                }
                break;
            }
            _ => break,
        }
    }
    ancestors
}

pub(super) fn ancestor_mass(inf: &Inferencer<'_, '_>, node: NodeId) -> Mass {
    source_mass(inf, &draw_sources(inf, node, &[]))
}

fn source_mass(inf: &Inferencer<'_, '_>, sources: &[NodeId]) -> Mass {
    sources.iter().fold(Mass::Normalized, |mass, &source| {
        combine_mass(mass, mass_of(inf, source).unwrap_or(Mass::Deferred))
    })
}

fn combine_mass(left: Mass, right: Mass) -> Mass {
    match (left, right) {
        (Mass::Normalized, Mass::Normalized) => Mass::Normalized,
        (Mass::Normalized | Mass::Deferred, Mass::Normalized | Mass::Deferred) => Mass::Deferred,
        _ => Mass::Unknown,
    }
}

/// Return source measures once per draw node, within the reification boundary.
fn draw_sources(inf: &Inferencer<'_, '_>, root: NodeId, boundary: &[Symbol]) -> Vec<NodeId> {
    let mut scopes = vec![boundary.to_vec()];
    let mut pending = vec![(root, 0)];
    let mut visited = HashSet::new();
    let mut draws = HashSet::new();
    let mut sources = Vec::new();
    while let Some((node, scope)) = pending.pop() {
        if inf.is_substituted(node) || !visited.insert((node, scope)) {
            continue;
        }
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if !scopes[scope].contains(&r.name)
                    && let Some(binding) = inf.module.binding_by_name(r.name)
                {
                    pending.push((inf.module.binding(binding).rhs, scope));
                }
            }
            Node::Call(call) => {
                let name = match call.head {
                    CallHead::Builtin(head) => inf.module.resolve(head),
                    _ => "",
                };
                if name == "draw" {
                    if let Some(&measure) = call.args.first() {
                        if draws.insert(node) {
                            sources.push(measure);
                        }
                        pending.push((measure, scope));
                    }
                    continue;
                }
                if matches!(name, "lawof" | "kernelof" | "elementof")
                    || matches!(inf.lookup_phase(node), Some(p) if p != Phase::Stochastic)
                {
                    continue;
                }
                let scope = if call.inputs.is_some() {
                    let mut boundary = scopes[scope].clone();
                    boundary.extend(super::reification_boundary_targets(inf, node));
                    boundary.sort_unstable();
                    boundary.dedup();
                    match scopes.iter().position(|existing| *existing == boundary) {
                        Some(index) => index,
                        None => {
                            scopes.push(boundary);
                            scopes.len() - 1
                        }
                    }
                } else {
                    scope
                };
                pending.extend(
                    inf.module
                        .node(node)
                        .children()
                        .into_iter()
                        .map(|n| (n, scope)),
                );
            }
            _ => {}
        }
    }
    sources
}
