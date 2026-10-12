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
            let Node::Call(call) = module.node(binding.rhs) else {
                return None;
            };
            let &law = call.args.first()?;
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
            Some((
                name.to_owned(),
                PriorFactor {
                    mass: mass.into(),
                    contains_flat: contains_flat(module, law),
                },
            ))
        })
        .collect()
}

fn contains_flat(module: &Module, root: NodeId) -> bool {
    let mut pending = vec![(root, false)];
    let mut visited = HashSet::new();
    while let Some((node, trace_values)) = pending.pop() {
        if !visited.insert((node, trace_values)) {
            continue;
        }
        match module.node(node) {
            Node::Ref(reference) => match reference.ns {
                RefNs::Module(alias)
                    if module.resolve(alias) == "reg"
                        && module.resolve(reference.name) == "Flat" =>
                {
                    return true;
                }
                RefNs::SelfMod => {
                    if let Some(binding) = module.binding_by_name(reference.name) {
                        pending.push((module.binding(binding).rhs, trace_values));
                    }
                }
                _ => {}
            },
            Node::Call(call) => {
                if let CallHead::User(callee) = call.head {
                    pending.push((callee, false));
                }
                // Only reified laws include the scalar value's draw ancestors.
                // A Normal's scale parameter does not change its own prior source.
                let trace_values = trace_values
                    || matches!(call.head, CallHead::Builtin(head)
                        if matches!(module.resolve(head), "lawof" | "kernelof"));
                for child in module.node(node).children() {
                    if trace_values
                        || matches!(
                            module.type_of(child),
                            Some(Type::Measure { .. } | Type::Kernel { .. })
                        )
                    {
                        pending.push((child, trace_values));
                    }
                }
            }
            _ => {}
        }
    }
    false
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
