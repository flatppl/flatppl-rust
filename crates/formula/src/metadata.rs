use crate::{
    FactorInfo, GroupInfo, ModelInfo, Parameter, ParameterRole, PredictorRecipe, PriorFactor,
    Result,
};
use flatppl_convert::{Builder, check_binding_name, sanitize_ident};
use flatppl_core::{CallHead, Mass, Module, Node, NodeId, RefNs, Type};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub(crate) fn levels(
    recipe: &PredictorRecipe,
) -> (BTreeMap<String, FactorInfo>, BTreeMap<String, GroupInfo>) {
    let factors: BTreeMap<_, _> = std::iter::once(&recipe.design)
        .chain(recipe.groups.iter().map(|group| &group.design))
        .flat_map(|design| &design.predictors)
        .filter_map(|predictor| {
            predictor.factor.as_ref().map(|factor| {
                (
                    predictor.column.clone(),
                    FactorInfo {
                        name: predictor.column.clone(),
                        levels: factor.levels.clone(),
                        ordered: factor.ordered,
                    },
                )
            })
        })
        .collect();
    let groups: BTreeMap<_, _> = recipe
        .groups
        .iter()
        .map(|group| {
            (
                group.column.clone(),
                GroupInfo {
                    name: group.column.clone(),
                    levels: group.levels.clone(),
                },
            )
        })
        .collect();
    (field_keys(factors), field_keys(groups))
}

// Metadata keys must survive printing even when R labels contain punctuation.
// Each value retains its original name; suffixes distinguish sanitized collisions.
fn field_keys<T>(values: BTreeMap<String, T>) -> BTreeMap<String, T> {
    let mut used = HashSet::new();
    values
        .into_iter()
        .map(|(name, value)| {
            let mut base = sanitize_ident(&name);
            if check_binding_name(&base, "metadata").is_err() {
                base = format!("field_{base}");
            }
            let mut key = base.clone();
            let mut suffix = 2;
            while !used.insert(key.clone()) {
                key = format!("{base}_{suffix}");
                suffix += 1;
            }
            (key, value)
        })
        .collect()
}

pub(crate) fn prior_factors(
    module: &Module,
    parameters: &[Parameter],
) -> BTreeMap<String, PriorFactor> {
    let sampled: HashSet<_> = parameters
        .iter()
        .filter(|parameter| parameter.role == ParameterRole::Sampled)
        .map(|parameter| parameter.binding.as_str())
        .collect();
    module
        .public_bindings()
        .filter_map(|(_, binding)| {
            let name = module.resolve(binding.name);
            if !sampled.contains(name) {
                return None;
            }
            let law =
                parameter_measure(module, binding.rhs).or_else(|| explicit_factor(module, name))?;
            let mass = match module.type_of(law) {
                Some(Type::Measure { mass, .. }) => *mass,
                _ => Mass::Unknown,
            };
            let mass = match mass {
                Mass::Deferred => "deferred",
                Mass::Null => "null",
                Mass::Normalized => "normalized",
                Mass::Finite => "finite",
                Mass::LocallyFinite => "locallyfinite",
                Mass::Unknown => "unknown",
            };
            Some((name.to_owned(), PriorFactor { mass: mass.into() }))
        })
        .collect()
}

/// Proper parameters carry their source in the draw declaration.
fn parameter_measure(module: &Module, node: NodeId) -> Option<NodeId> {
    let Node::Call(call) = module.node(node) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    match module.resolve(head) {
        "draw" => call.args.first().copied(),
        _ => None,
    }
}

/// Explicit factors live in a named joint, possibly under conditional layers.
fn explicit_factor(module: &Module, name: &str) -> Option<NodeId> {
    let (_, prior) = module
        .public_bindings()
        .find(|(_, binding)| module.resolve(binding.name) == "Pi_prior")?;
    let mut pending = vec![prior.rhs];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        match module.node(node) {
            Node::Call(call) if matches!(call.head, CallHead::Builtin(head) if matches!(module.resolve(head), "joint" | "jointchain")) =>
            {
                if let Some(field) = call
                    .named
                    .iter()
                    .find(|field| module.resolve(field.name) == name)
                {
                    // A later component of a chain is a kernel: its factor is the body.
                    return Some(kernel_body(module, field.value));
                }
                pending.extend(module.node(node).children());
            }
            Node::Ref(reference) if reference.ns == RefNs::SelfMod => {
                if let Some(binding) = module.binding_by_name(reference.name) {
                    pending.push(module.binding(binding).rhs);
                }
            }
            _ => pending.extend(module.node(node).children()),
        }
    }
    None
}

/// The measure a chain component's kernel returns, or the component itself.
fn kernel_body(module: &Module, node: NodeId) -> NodeId {
    match module.node(node) {
        Node::Call(call)
            if matches!(call.head, CallHead::Builtin(head) if module.resolve(head) == "functionof")
                && call.args.len() == 1 =>
        {
            call.args[0]
        }
        _ => node,
    }
}

pub(crate) fn emit(module: &mut Module, info: &ModelInfo) -> Result<()> {
    let mut names: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    for parameter in &info.parameters {
        let indices = parameter
            .indices
            .clone()
            .unwrap_or_else(|| parameter.index.into_iter().collect());
        let key = if indices.is_empty() {
            "scalar".into()
        } else {
            indices.iter().map(|index| format!("_{index}")).collect()
        };
        names
            .entry(parameter.binding.clone())
            .or_default()
            .insert(key, serde_json::to_value(parameter)?);
    }
    let names = serde_json::to_value(names)?;
    let mut model = serde_json::to_value(info)?;
    model["parameters"] = names.clone();
    let mut b = Builder::new(module);
    let names = literal(&mut b, &names)?;
    b.bind("formula_parameter_names", names);
    let model = literal(&mut b, &model)?;
    b.bind("formula_model", model);
    Ok(())
}

fn literal(b: &mut Builder<'_>, value: &Value) -> Result<NodeId> {
    Ok(match value {
        Value::Bool(value) => b.lit_bool(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                b.lit_int(value)
            } else {
                b.lit_real(value.as_f64().unwrap())
            }
        }
        Value::String(value) => b.str_lit(value),
        Value::Array(values) => {
            // Indexed records also represent empty and structured sequences.
            let items = values
                .iter()
                .map(|value| literal(b, value))
                .collect::<Result<Vec<_>>>()?;
            let names: Vec<_> = (1..=items.len()).map(|i| format!("_{i}")).collect();
            let mut fields = vec![("length", b.lit_int(items.len() as i64))];
            fields.extend(names.iter().map(String::as_str).zip(items));
            b.call_fields("record", &fields)
        }
        Value::Object(fields) => {
            let integer_level = fields.get("kind").and_then(Value::as_str) == Some("integer");
            let fields = fields
                .iter()
                .map(|(name, value)| {
                    // Full-width labels must remain readable without arithmetic.
                    let value = if integer_level && name == "value" {
                        b.str_lit(&value.to_string())
                    } else {
                        literal(b, value)?
                    };
                    Ok((name.as_str(), value))
                })
                .collect::<Result<Vec<_>>>()?;
            b.call_fields("record", &fields)
        }
        Value::Null => return Err(crate::Error("formula metadata contains null".into())),
    })
}
