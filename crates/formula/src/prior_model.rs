//! Compose proper laws and explicit measure factors without improper draws.
use crate::{Error, Result, emit::parameter_function};
use flatppl_convert::Builder;
use flatppl_core::{CallHead, Dim, Mass, Module, Node, NodeId, RefNs, Type};
use std::collections::HashSet;

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
    let mut done = HashSet::new();
    let mut chain = Vec::new();
    while done.len() < factors.len() {
        let ready: Vec<_> = factors
            .iter()
            .enumerate()
            .filter(|(i, factor)| !done.contains(i) && factor.dependencies.is_subset(&done))
            .collect();
        if ready.is_empty() {
            return Err(Error("prior parameters have cyclic dependencies".into()));
        }
        let measure = group(b, ready.iter().map(|(_, factor)| *factor).collect(), false);
        let component = if chain.is_empty() {
            measure
        } else {
            let inputs = names
                .iter()
                .enumerate()
                .filter(|(i, _)| done.contains(i))
                .map(|(_, name)| name.clone())
                .collect::<Vec<_>>();
            let kernel = parameter_function(b, "functionof", measure, &inputs);
            if inputs.len() == 1 {
                b.record_adapter(kernel, &inputs)
            } else {
                kernel
            }
        };
        chain.push(component);
        done.extend(ready.into_iter().map(|(i, _)| i));
    }
    Ok(b.call("jointchain", &chain))
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
