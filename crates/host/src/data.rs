//! Host-side data loading (spec §07 `load_data`). The declared domain determines
//! shape and selection; [`crate::readers`] decodes the source by its extension.

use std::collections::{HashMap, HashSet};

use flatppl_core::{CallHead, Dim, Module, Node, NodeId, RefNs, Scalar, ScalarType, Symbol, Type};

use crate::Constant;
use crate::compiler::resolve_ref;
use crate::readers::{Datum, decode, format};

/// A `load_data` source that cannot be loaded, localized to its call when the
/// call is known.
#[derive(Debug)]
pub struct DataError {
    /// The `load_data` call, in the module passed to [`load_fixed_data`].
    pub node: Option<NodeId>,
    /// The source as the call names it, after path resolution.
    pub source: Option<String>,
    pub message: String,
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            Some(source) => write!(f, "load_data({source:?}): {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl From<String> for DataError {
    fn from(message: String) -> Self {
        DataError {
            node: None,
            source: None,
            message,
        }
    }
}

impl From<&str> for DataError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

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
/// Scalars, arrays, records and tables are read from JSON, CSV, WSV, Arrow IPC
/// and Safetensors sources (§07). `read` retains the caller's file-access/cache
/// policy. Runtime inputs are never read.
pub fn load_fixed_data(
    module: &mut Module,
    mut read: impl FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<(), DataError> {
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
            let fail = |message: String| DataError {
                node: Some(node),
                source: Some(source.to_string()),
                message,
            };
            let format = format(source).map_err(fail)?;
            let bytes = read(source).map_err(fail)?;
            let datum = decode(format, &bytes).map_err(fail)?;
            let ty = module
                .type_of(node)
                .ok_or_else(|| fail("unresolved data type".into()))?;
            let value = selected_value(module, &datum, ty)
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
            return Err(error.message.into());
        }
    }
    Ok(())
}

fn selected_value(module: &Module, datum: &Datum, ty: &Type) -> Result<Constant, String> {
    match ty {
        Type::Scalar(scalar) => scalar_value(datum, *scalar),
        Type::Array { shape, elem } => {
            let dims = shape
                .iter()
                .map(|dim| match dim {
                    Dim::Static(n) => Ok(*n),
                    Dim::Dynamic => Err("load_data dimensions must be fixed".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            // §07 lets a delimited or Arrow source with one column hold a vector.
            let datum = match datum {
                Datum::Columns(columns) if dims.len() == 1 && columns.len() == 1 => &columns[0].1,
                Datum::Columns(columns) if dims.len() == 1 => {
                    return Err(format!(
                        "a vector reads a source with exactly one column, and this one has {}",
                        columns.len()
                    ));
                }
                datum => datum,
            };
            let mut data = Vec::new();
            match datum {
                Datum::Tensor {
                    shape,
                    data: leaves,
                } => tensor_prefix(module, shape, leaves, &dims, elem, &mut data)?,
                datum => array_prefix(module, datum, &dims, elem, &mut data)?,
            }
            Ok(Constant::Array { shape: dims, data })
        }
        Type::Record(fields) => fields
            .iter()
            .map(|(name, ty)| {
                let name = module.resolve(*name);
                let field = datum
                    .field(name)
                    .ok_or_else(|| format!("missing field `{name}`"))?;
                Ok((name.into(), selected_value(module, field, ty)?))
            })
            .collect::<Result<_, _>>()
            .map(Constant::Record),
        Type::Table { columns, nrows } => {
            // §07 JSON: struct-of-arrays, or array-of-structs read row by row.
            let rows = match datum {
                Datum::List(rows) => Some(rows),
                _ => None,
            };
            columns
                .iter()
                .map(|(name, ty)| {
                    let name = module.resolve(*name);
                    let transposed;
                    let field = match rows {
                        Some(rows) => {
                            transposed = Datum::List(
                                rows.iter()
                                    .enumerate()
                                    .map(|(i, row)| {
                                        row.field(name).cloned().ok_or_else(|| {
                                            format!("row {} is missing field `{name}`", i + 1)
                                        })
                                    })
                                    .collect::<Result<_, _>>()?,
                            );
                            &transposed
                        }
                        None => datum
                            .field(name)
                            .ok_or_else(|| format!("missing column `{name}`"))?,
                    };
                    let column = Type::Array {
                        shape: vec![*nrows].into(),
                        elem: Box::new(ty.clone()),
                    };
                    Ok((name.into(), selected_value(module, field, &column)?))
                })
                .collect::<Result<_, _>>()
                .map(Constant::Table)
        }
        _ => Err("load_data needs a declared numeric scalar, array, record or table domain".into()),
    }
}

/// A scalar of `kind`. Typed sources keep their own kind for the domain check; a
/// delimited cell is text, parsed here once its set is known.
fn scalar_value(datum: &Datum, kind: ScalarType) -> Result<Constant, String> {
    // §03 nests booleans ⊂ integers ⊂ reals, so only a wider source kind refuses.
    match (datum, kind) {
        (Datum::Real(_) | Datum::Real32(_), ScalarType::Integer | ScalarType::Boolean) => {
            return Err("a real value where the declared set holds integers or booleans".into());
        }
        (Datum::Int(_), ScalarType::Boolean) => {
            return Err("an integer value where the declared set holds booleans".into());
        }
        _ => {}
    }
    match datum {
        Datum::Bool(b) => Ok(Constant::Bool(*b)),
        Datum::Int(i) => Ok(Constant::Integer(*i)),
        Datum::Real(x) => Ok(Constant::Real(*x)),
        Datum::Real32(x) => Ok(Constant::Real32(*x)),
        Datum::Null => Err("a value is missing".into()),
        Datum::Tensor { shape, data } if shape.is_empty() => scalar_value(&data[0], kind),
        Datum::Text(cell) => {
            let parsed = match kind {
                ScalarType::Boolean if cell.eq_ignore_ascii_case("true") => {
                    Some(Constant::Bool(true))
                }
                ScalarType::Boolean if cell.eq_ignore_ascii_case("false") => {
                    Some(Constant::Bool(false))
                }
                ScalarType::Boolean => None,
                ScalarType::Integer => cell.parse::<i64>().map(Constant::Integer).ok(),
                _ => cell.parse::<i64>().map(Constant::Integer).ok().or_else(|| {
                    cell.parse::<f64>()
                        .ok()
                        .filter(|x| !x.is_nan())
                        .map(Constant::Real)
                }),
            };
            parsed.ok_or_else(|| format!("cell `{cell}` is not a value of its declared set"))
        }
        _ => Err("expected a scalar".into()),
    }
}

fn array_prefix(
    module: &Module,
    datum: &Datum,
    shape: &[u32],
    elem: &Type,
    data: &mut Vec<Constant>,
) -> Result<(), String> {
    let Some((&n, rest)) = shape.split_first() else {
        data.push(selected_value(module, datum, elem)?);
        return Ok(());
    };
    let Datum::List(values) = datum else {
        return Err("expected an array".into());
    };
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

/// A Safetensors tensor as the declared array: the shape must match, except that a
/// vector may read its first `n` entries (§07 "the first `n` entries").
fn tensor_prefix(
    module: &Module,
    shape: &[usize],
    leaves: &[Datum],
    dims: &[u32],
    elem: &Type,
    data: &mut Vec<Constant>,
) -> Result<(), String> {
    let fits = shape.len() == dims.len()
        && shape
            .iter()
            .zip(dims)
            .enumerate()
            .all(|(axis, (&have, &want))| {
                have == want as usize || (dims.len() == 1 && axis == 0 && have >= want as usize)
            });
    if !fits {
        return Err(format!(
            "tensor shape {shape:?} does not match the declared shape {dims:?}"
        ));
    }
    let count = dims.iter().map(|&d| d as usize).product::<usize>();
    for leaf in &leaves[..count] {
        data.push(selected_value(module, leaf, elem)?);
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
