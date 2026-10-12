//! Host-side JSON data loading. Declared domains determine shape and selection.

use std::collections::{HashMap, HashSet};

use flatppl_core::{CallHead, Dim, Module, Node, NodeId, RefNs, Scalar, Symbol, Type};
use serde_json::Value;

use crate::Constant;
use crate::compiler::resolve_ref;

/// Resolve data source names before a module is grafted into its importer.
/// The caller supplies the declaring file's path and bundle policy; no data is read.
pub fn resolve_data_sources(
    module: &mut Module,
    mut resolve: impl FnMut(&str) -> Result<String, String>,
) -> Result<(), String> {
    rewrite(module, &mut |module, node| {
        if !builtin(module, node, "load_data") {
            return Ok(None);
        }
        let Node::Call(mut call) = module.node(node).clone() else {
            unreachable!()
        };
        let source = call.args.first().copied().or_else(|| {
            call.named
                .iter()
                .find(|arg| module.resolve(arg.name) == "source")
                .map(|arg| arg.value)
        });
        let Some(source) = source else {
            return Ok(None);
        };
        let Node::Lit(Scalar::Str(source)) = module.node(resolve_ref(module, source)) else {
            return Ok(None);
        };
        let resolved = resolve(source)?;
        let value = module.alloc(Node::Lit(Scalar::Str(resolved.into())));
        if let Some(first) = call.args.first_mut() {
            *first = value;
        } else if let Some(arg) = call
            .named
            .iter_mut()
            .find(|arg| module.resolve(arg.name) == "source")
        {
            arg.value = value;
        }
        Ok(Some(module.alloc(Node::Call(call))))
    })
}

/// Bake reached, unpromoted `load_data` values into a determinized module.
/// Numeric JSON scalars, arrays, records and column-oriented tables are supported.
/// `read` retains the caller's file-access/cache policy. Runtime inputs are never read.
pub fn load_fixed_data(
    module: &mut Module,
    mut read: impl FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<(), String> {
    let inputs: HashSet<Symbol> = binding(module, "inputs")
        .map(|node| signature(module, node))
        .unwrap_or_default()
        .into_iter()
        .filter_map(|node| match module.node(node) {
            Node::Ref(reference) if reference.ns == RefNs::SelfMod => Some(reference.name),
            _ => None,
        })
        .collect();
    let mut pending = binding(module, "outputs")
        .map(|node| vec![node])
        .unwrap_or_else(|| {
            module
                .bindings()
                .filter(|(_, b)| !inputs.contains(&b.name))
                .map(|(_, b)| b.rhs)
                .collect()
        });
    let mut seen = HashSet::new();
    let mut replacements = HashMap::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Node::Ref(reference) = module.node(node) {
            if reference.ns == RefNs::SelfMod
                && !inputs.contains(&reference.name)
                && let Some(binding) = module.binding_by_name(reference.name)
            {
                pending.push(module.binding(binding).rhs);
            }
        } else if builtin(module, node, "load_data") {
            let Node::Call(call) = module.node(node) else {
                unreachable!()
            };
            let source = call
                .args
                .first()
                .copied()
                .or_else(|| {
                    call.named
                        .iter()
                        .find(|arg| module.resolve(arg.name) == "source")
                        .map(|arg| arg.value)
                })
                .ok_or("load_data requires a source")?;
            let Node::Lit(Scalar::Str(source)) = module.node(resolve_ref(module, source)) else {
                return Err("load_data requires a fixed source string".into());
            };
            let fail = |message| format!("load_data({source:?}): {message}");
            let json: Value = serde_json::from_slice(&read(source).map_err(fail)?)
                .map_err(|error| fail(error.to_string()))?;
            let ty = module
                .type_of(node)
                .ok_or_else(|| fail("unresolved data type".into()))?;
            let value = selected_value(module, &json, ty)
                .map_err(fail)?
                .validated(module, node)
                .map_err(fail)?;
            let value = value.alloc(module)?;
            replacements.insert(node, value);
        } else {
            pending.extend(module.node(node).children());
        }
    }
    if !replacements.is_empty() {
        rewrite(module, &mut |_, node| Ok(replacements.get(&node).copied()))?;
        if let Some(error) = flatppl_infer::infer(module)
            .into_iter()
            .find(|error| error.severity == flatppl_infer::Severity::Error)
        {
            return Err(error.message);
        }
    }
    Ok(())
}

fn selected_value(module: &Module, json: &Value, ty: &Type) -> Result<Constant, String> {
    match ty {
        Type::Scalar(_) => match json {
            Value::Bool(value) => Ok(Constant::Bool(*value)),
            Value::Number(value) => value
                .as_i64()
                .map(Constant::Integer)
                .or_else(|| value.as_f64().map(Constant::Real))
                .ok_or_else(|| "number is not representable".into()),
            _ => Err("expected a numeric scalar".into()),
        },
        Type::Array { shape, elem } => {
            let dims = shape
                .iter()
                .map(|dim| match dim {
                    Dim::Static(n) => Ok(*n),
                    Dim::Dynamic => Err("load_data dimensions must be fixed".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut data = Vec::new();
            array_prefix(module, json, &dims, elem, &mut data)?;
            Ok(Constant::Array { shape: dims, data })
        }
        Type::Record(fields) => fields
            .iter()
            .map(|(name, ty)| {
                let name = module.resolve(*name);
                let field = json
                    .get(name)
                    .ok_or_else(|| format!("missing field `{name}`"))?;
                Ok((name.into(), selected_value(module, field, ty)?))
            })
            .collect::<Result<_, _>>()
            .map(Constant::Record),
        Type::Table { columns, nrows } => columns
            .iter()
            .map(|(name, ty)| {
                let name = module.resolve(*name);
                let field = json
                    .get(name)
                    .ok_or_else(|| format!("missing column `{name}`"))?;
                let column = Type::Array {
                    shape: vec![*nrows].into(),
                    elem: Box::new(ty.clone()),
                };
                Ok((name.into(), selected_value(module, field, &column)?))
            })
            .collect::<Result<_, _>>()
            .map(Constant::Table),
        _ => Err("load_data needs a declared numeric scalar, array, record or table domain".into()),
    }
}

fn array_prefix(
    module: &Module,
    json: &Value,
    shape: &[u32],
    elem: &Type,
    data: &mut Vec<Constant>,
) -> Result<(), String> {
    let Some((&n, rest)) = shape.split_first() else {
        data.push(selected_value(module, json, elem)?);
        return Ok(());
    };
    let values = json.as_array().ok_or("expected a JSON array")?;
    if values.len() < n as usize {
        return Err(format!(
            "requested {n} entries but only {} are available",
            values.len()
        ));
    }
    for value in &values[..n as usize] {
        array_prefix(module, value, rest, elem, data)?;
    }
    Ok(())
}

fn binding(module: &Module, name: &str) -> Option<NodeId> {
    module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)
        .map(|(_, b)| b.rhs)
}

fn builtin(module: &Module, node: NodeId, name: &str) -> bool {
    matches!(module.node(node), Node::Call(call) if matches!(call.head, CallHead::Builtin(head) if module.resolve(head) == name))
}

fn signature(module: &Module, node: NodeId) -> Vec<NodeId> {
    if builtin(module, node, "tuple")
        && let Node::Call(call) = module.node(node)
    {
        call.args.to_vec()
    } else {
        vec![node]
    }
}

fn rewrite(
    module: &mut Module,
    replace: &mut impl FnMut(&mut Module, NodeId) -> Result<Option<NodeId>, String>,
) -> Result<(), String> {
    fn visit(
        module: &mut Module,
        node: NodeId,
        replace: &mut impl FnMut(&mut Module, NodeId) -> Result<Option<NodeId>, String>,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> Result<NodeId, String> {
        if let Some(&mapped) = memo.get(&node) {
            return Ok(mapped);
        }
        let mapped = if let Some(mapped) = replace(module, node)? {
            mapped
        } else if let Node::Call(mut call) = module.node(node).clone() {
            if let CallHead::User(callee) = &mut call.head {
                *callee = visit(module, *callee, replace, memo)?;
            }
            for arg in call.args.iter_mut() {
                *arg = visit(module, *arg, replace, memo)?;
            }
            for arg in call.named.iter_mut() {
                arg.value = visit(module, arg.value, replace, memo)?;
            }
            if module.node(node) == &Node::Call(call.clone()) {
                node
            } else {
                module.alloc(Node::Call(call))
            }
        } else {
            node
        };
        if mapped != node
            && let Some(span) = module.span_of(node)
        {
            module.set_span(mapped, span);
        }
        memo.insert(node, mapped);
        Ok(mapped)
    }
    let roots: Vec<_> = module
        .bindings()
        .map(|(id, binding)| (id, binding.rhs))
        .collect();
    let mut memo = HashMap::new();
    for (binding, node) in roots {
        let rhs = visit(module, node, replace, &mut memo)?;
        module.set_binding_rhs(binding, rhs);
    }
    Ok(())
}
