//! Ordinary call interfaces, independent of argument domains.

use flatppl_core::{CallHead, Inputs, Node, NodeId, RefNs, Scalar, Type};

use crate::catalogue::{Arity, Sig, builtin, sig_arity, sig_param_names};
use crate::trace::Inferencer;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Interface {
    pub names: Vec<String>,
    pub arity: Arity,
    pub ordered: bool,
}

impl Interface {
    pub fn takes_one(&self) -> bool {
        self.ordered && self.arity.admits(1)
    }

    pub fn required(names: Vec<String>, ordered: bool) -> Self {
        Self {
            arity: Arity {
                min: names.len(),
                max: Some(names.len()),
            },
            names,
            ordered,
        }
    }

    pub fn from_sig(sig: &Sig) -> Option<Self> {
        Some(Self {
            names: sig_param_names(sig).unwrap_or_default().to_vec(),
            arity: sig_arity(sig)?,
            ordered: true,
        })
    }
}

pub(crate) fn interface(inf: &Inferencer<'_, '_>, mut node: NodeId) -> Option<Interface> {
    let m = &*inf.module;
    for _ in 0..m.node_count() {
        if let Some(interface) = inf.module_callable_interface(node) {
            return Some(interface.clone());
        }
        if let Some(reference) = inf.module_catalogue_ref(node) {
            return Interface::from_sig(&reference.sig);
        }
        match m.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                node = m.binding(m.binding_by_name(r.name)?).rhs;
            }
            Node::Const(name) => {
                let name = m.resolve(*name);
                return Some(Interface {
                    names: builtin()
                        .base_param_names(name)
                        .unwrap_or_default()
                        .to_vec(),
                    arity: builtin().base_arity(name)?,
                    ordered: true,
                });
            }
            Node::Call(call) => {
                if let Some(inputs) = &call.inputs {
                    let (entries, ordered) = match inputs {
                        Inputs::Spec(entries) => (entries.as_ref(), true),
                        Inputs::Auto => (inf.auto_inputs_of(node)?, false),
                    };
                    return Some(Interface::required(
                        entries
                            .iter()
                            .map(|(name, _)| m.resolve(*name).to_string())
                            .collect(),
                        ordered,
                    ));
                }
                let CallHead::Builtin(head) = call.head else {
                    node = forwarded_result(inf, node)?;
                    continue;
                };
                match m.resolve(head) {
                    "fchain" | "broadcasted" | "bijection" | "ksuperpose" => {
                        node = *call.args.first()?;
                    }
                    "get" | "get0" => {
                        let mut source = *call.args.first()?;
                        for _ in 0..m.binding_count() {
                            let Node::Ref(r) = m.node(source) else { break };
                            if r.ns != RefNs::SelfMod {
                                break;
                            }
                            source = m.binding(m.binding_by_name(r.name)?).rhs;
                        }
                        let Node::Call(value) = m.node(source) else {
                            return None;
                        };
                        let CallHead::Builtin(op) = value.head else {
                            return None;
                        };
                        let Node::Lit(Scalar::Int(index)) = m.node(*call.args.get(1)?) else {
                            return None;
                        };
                        let index = index.checked_sub(i64::from(m.resolve(head) == "get"))?;
                        if m.resolve(op) == "tuple" {
                            node = *value.args.get(usize::try_from(index).ok()?)?;
                        } else if m.resolve(op) == "disintegrate" && index == 0 {
                            let Type::Kernel { inputs, .. } = inf.lookup_type(node)? else {
                                return None;
                            };
                            if inputs.is_empty() {
                                return None;
                            }
                            let Type::Measure { domain, .. } =
                                inf.lookup_type(*value.args.get(1)?)?
                            else {
                                return None;
                            };
                            return Some(Interface::required(
                                inputs
                                    .iter()
                                    .map(|name| m.resolve(*name).to_string())
                                    .collect(),
                                !matches!(domain.as_ref(), Type::Record(_)),
                            ));
                        } else {
                            return None;
                        }
                    }
                    "joint" => {
                        let Type::Kernel { inputs, .. } = inf.lookup_type(node)? else {
                            return None;
                        };
                        return Some(Interface::required(
                            inputs
                                .iter()
                                .map(|name| m.resolve(*name).to_string())
                                .collect(),
                            false,
                        ));
                    }
                    // Uniform kernel extension preserves the underlying interface.
                    "normalize" | "truncate" | "locscale" | "iid" | "relabel" | "pushfwd"
                    | "weighted" | "logweighted" | "likelihoodof" => {
                        let index = usize::from(matches!(
                            m.resolve(head),
                            "pushfwd" | "weighted" | "logweighted"
                        ));
                        node = *call.args.get(index)?;
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
    None
}

/// Resolve a helper that returns an existing callable, without changing its
/// interface or constructing a new reification.
pub(crate) fn forwarded_result(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<NodeId> {
    let m = &*inf.module;
    let Node::Call(call) = m.node(node) else {
        return None;
    };
    let CallHead::User(mut callee) = call.head else {
        return None;
    };
    for _ in 0..m.binding_count() {
        let Node::Ref(r) = m.node(callee) else { break };
        if r.ns != RefNs::SelfMod {
            break;
        }
        callee = m.binding(m.binding_by_name(r.name)?).rhs;
    }
    let Node::Call(reified) = m.node(callee) else {
        return None;
    };
    let entries = match reified.inputs.as_ref()? {
        Inputs::Spec(entries) => entries.as_ref(),
        Inputs::Auto => inf.auto_inputs_of(callee)?,
    };
    let body = *reified.args.first()?;
    if let Node::Ref(reference) = m.node(body)
        && let Some(index) = entries.iter().position(|(_, target)| target == reference)
    {
        call.named
            .iter()
            .find(|arg| arg.name == entries[index].0)
            .map(|arg| arg.value)
            .or_else(|| call.args.get(index).copied())
    } else {
        Some(body)
    }
}
