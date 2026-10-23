//! Compose proper laws and explicit measure factors without improper draws.
use crate::{Error, Result};
use flatppl_convert::Builder;
use flatppl_core::{
    Call, CallHead, Dim, Inputs, Mass, Module, NamedArg, Node, NodeId, Ref, RefNs, Type,
};
use std::collections::{HashMap, HashSet};

pub(crate) struct ParameterPrior {
    pub name: String,
    pub measure: NodeId,
}

struct Factor<'a> {
    name: &'a str,
    measure: NodeId,
    proper: bool,
    dependencies: HashSet<usize>,
}

pub(crate) fn build(b: &mut Builder<'_>, parameters: &[ParameterPrior]) -> Result<NodeId> {
    let mut factors = Vec::new();
    let names: Vec<_> = parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect();
    for parameter in parameters {
        let name = &parameter.name;
        let measure = parameter.measure;
        let (types, diagnostics) = flatppl_infer::infer_expression(b.m, measure, &[]);
        let Some(Type::Measure { mass, .. }) = types.get(&measure) else {
            return Err(Error(format!(
                "prior for `{name}` is not a measure{}",
                diagnostics
                    .first()
                    .map_or(String::new(), |d| format!(": {}", d.message))
            )));
        };
        factors.push(Factor {
            name,
            measure,
            proper: matches!(mass, Mass::Normalized | Mass::Deferred),
            dependencies: dependencies(b.m, measure, &names),
        });
    }
    for factor in &factors {
        if factor.proper {
            let symbol = b.sym(factor.name);
            let binding = b.m.binding_by_name(symbol).unwrap();
            let value = b.call("draw", &[factor.measure]);
            b.m.set_binding_rhs(binding, value);
        }
    }
    // Independent groups keep the normal workflow: one combined proper law,
    // joined to the explicit improper factors. Dependencies within a proper
    // law remain part of its shared trace.
    if factors.iter().all(|factor| {
        factor
            .dependencies
            .iter()
            .all(|&i| factors[i].proper == factor.proper)
    }) && factors
        .iter()
        .filter(|factor| !factor.proper)
        .all(|factor| factor.dependencies.is_empty())
    {
        return Ok(group(b, factors.iter().collect(), true));
    }

    // A reference factor may depend on a proper scale, or a proper law may
    // depend on an improper location. Bind the preceding coordinates explicitly
    // rather than relying on matching names to condition a measure product.
    // The keyword form of `jointchain` (spec §06 "Keyword form") labels each component's
    // variate by its parameter name, and each later kernel receives the record of the
    // preceding variates by the ordinary calling convention. A kernel of one input takes
    // a sole record whole (spec §04 "Calling conventions"), so every later kernel is a
    // function of that record and reads the preceding parameters from its fields.
    let mut done = HashSet::new();
    let mut chain: Vec<(&str, NodeId)> = Vec::new();
    while done.len() < factors.len() {
        let ready: Vec<_> = factors
            .iter()
            .enumerate()
            .filter(|(i, factor)| !done.contains(i) && factor.dependencies.is_subset(&done))
            .collect();
        if ready.is_empty() {
            return Err(Error("prior parameters have cyclic dependencies".into()));
        }
        for (_, factor) in &ready {
            let measure = if factor.proper {
                let value = b.self_ref(factor.name);
                b.call("lawof", &[value])
            } else {
                factor.measure
            };
            let component = if chain.is_empty() {
                measure
            } else {
                let preceding = chain
                    .iter()
                    .map(|(name, _)| (*name).to_string())
                    .collect::<Vec<_>>();
                record_kernel(b, measure, &preceding)
            };
            chain.push((factor.name, component));
        }
        done.extend(ready.into_iter().map(|(i, _)| i));
    }
    Ok(b.call_kw("jointchain", &chain))
}

/// `measure` as a function of the record of the `preceding` parameters: each reference to one
/// of them, directly or through a derived binding, reads the field of the record instead.
fn record_kernel(b: &mut Builder<'_>, measure: NodeId, preceding: &[String]) -> NodeId {
    let point = Ref {
        ns: RefNs::Local,
        name: b.sym(&format!("_point_{}_", b.m.node_count())),
    };
    let value = b.m.alloc(Node::Ref(point));
    let body = read_fields(b, measure, preceding, value, &mut HashMap::new());
    let head = b.sym("functionof");
    let label = b.sym("point");
    b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([body]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(Box::new([(label, point)]))),
    }))
}

fn read_fields(
    b: &mut Builder<'_>,
    node: NodeId,
    parameters: &[String],
    point: NodeId,
    seen: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&done) = seen.get(&node) {
        return done;
    }
    let rewritten = match b.m.node(node).clone() {
        Node::Ref(reference) if reference.ns == RefNs::SelfMod => {
            let name = b.m.resolve(reference.name).to_string();
            if parameters.contains(&name) {
                let field = b.str_lit(&name);
                b.call("get", &[point, field])
            } else if let Some(binding) = b.m.binding_by_name(reference.name) {
                let rhs = b.m.binding(binding).rhs;
                if dependencies(b.m, rhs, parameters).is_empty() {
                    node
                } else {
                    read_fields(b, rhs, parameters, point, seen)
                }
            } else {
                node
            }
        }
        Node::Call(call) => {
            let head = match call.head {
                CallHead::User(callee) => {
                    CallHead::User(read_fields(b, callee, parameters, point, seen))
                }
                head => head,
            };
            let args: Vec<_> = call
                .args
                .iter()
                .map(|&arg| read_fields(b, arg, parameters, point, seen))
                .collect();
            let named: Vec<_> = call
                .named
                .iter()
                .map(|arg| NamedArg {
                    kind: arg.kind,
                    name: arg.name,
                    value: read_fields(b, arg.value, parameters, point, seen),
                })
                .collect();
            let unchanged = args.iter().eq(call.args.iter())
                && named
                    .iter()
                    .zip(call.named.iter())
                    .all(|(a, c)| a.value == c.value)
                && head == call.head;
            if unchanged {
                node
            } else {
                b.m.alloc(Node::Call(Call {
                    head,
                    args: args.into(),
                    named: named.into(),
                    inputs: call.inputs.clone(),
                }))
            }
        }
        _ => node,
    };
    seen.insert(node, rewritten);
    rewritten
}

fn group(b: &mut Builder<'_>, factors: Vec<&Factor<'_>>, named: bool) -> NodeId {
    let mut proper = Vec::new();
    let mut improper = Vec::new();
    for factor in factors {
        if factor.proper {
            proper.push((factor.name, b.self_ref(factor.name)));
        } else {
            improper.push((factor.name, factor.measure));
        }
    }
    let mut measures = Vec::new();
    if !improper.is_empty() {
        let measure = b.call_kw("joint", &improper);
        measures.push(if named {
            b.bind("Pi_improper", measure);
            b.self_ref("Pi_improper")
        } else {
            measure
        });
    }
    if !proper.is_empty() {
        let record = b.call_fields("record", &proper);
        let measure = b.call("lawof", &[record]);
        measures.push(if named && !measures.is_empty() {
            b.bind("Pi_proper", measure);
            b.self_ref("Pi_proper")
        } else {
            measure
        });
    }
    if measures.len() == 1 {
        measures[0]
    } else {
        b.call("joint", &measures)
    }
}

fn dependencies(module: &Module, root: NodeId, parameters: &[String]) -> HashSet<usize> {
    let mut result = HashSet::new();
    let mut seen = HashSet::new();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Node::Ref(reference) = module.node(node)
            && reference.ns == RefNs::SelfMod
        {
            if let Some(i) = parameters
                .iter()
                .position(|name| name == module.resolve(reference.name))
            {
                result.insert(i);
            } else if let Some(binding) = module.binding_by_name(reference.name) {
                pending.push(module.binding(binding).rhs);
            }
        } else {
            pending.extend(module.node(node).children());
        }
    }
    result
}

/// Keep constraints from an explicit reference, including live interval bounds.
/// Other measure supports may be conservative, so their density still gates the
/// point. Provisional proper coordinates become draws before source emission.
pub(crate) fn parameter_support(b: &mut Builder<'_>, measure: NodeId) -> NodeId {
    if let Some(support) = crate::prior_expr::reference_support(b.m, measure) {
        return support;
    }
    if let Node::Call(call) = b.m.node(measure).clone()
        && let CallHead::Builtin(head) = call.head
    {
        match b.m.resolve(head) {
            "truncate" if call.args.len() == 2 => {
                return call.args[1];
            }
            "iid" if call.args.len() == 2 => {
                let support = parameter_support(b, call.args[0]);
                return b.call("cartpow", &[support, call.args[1]]);
            }
            "weighted" | "logweighted" if call.args.len() == 2 => {
                return parameter_support(b, call.args[1]);
            }
            _ => {}
        }
    }
    let (types, _) = flatppl_infer::infer_expression(b.m, measure, &[]);
    let domain = match types.get(&measure) {
        Some(Type::Measure { domain, .. }) => domain.as_ref(),
        _ => &Type::Deferred,
    };
    natural_set(b, domain)
}

fn natural_set(b: &mut Builder<'_>, domain: &Type) -> NodeId {
    let reals = b.call_head("reals");
    if let Type::Array { shape, elem } = domain {
        let support = natural_set(b, elem);
        let dimensions: Option<Vec<_>> = shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(n) => Some(b.lit_int(i64::from(*n))),
                Dim::Dynamic => None,
            })
            .collect();
        if let Some(dimensions) = dimensions {
            let size = if dimensions.len() == 1 {
                dimensions[0]
            } else {
                b.array(&dimensions)
            };
            return b.call("cartpow", &[support, size]);
        }
    }
    reals
}
