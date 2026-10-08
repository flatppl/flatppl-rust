//! Explicit query ABI and host metadata, using the existing StableHLO builder.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    Call, CallHead, Dim, Idx, Module, Node, NodeId, Phase, RefNs, Scalar, Symbol, Type,
};
use flatppl_stablehlo::{
    BatchSpec, Dtype, ElemKind, EmitError, EmitOptions, Emitter, MlirTy, Value, check_query_inputs,
    mlir_type_of,
};
use serde::Serialize;

#[derive(Serialize)]
pub struct Export {
    pub stablehlo: String,
    pub entry_point: &'static str,
    pub inputs: Vec<Field>,
    pub output: Schema,
    pub multiple_outputs: bool,
    pub output_names: Vec<Option<String>>,
    /// Host batch dimensions precede every result's authored cell dimensions.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub batch_shape: Vec<u64>,
}

#[derive(Serialize)]
pub struct Field {
    pub name: String,
    pub value: Schema,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schema {
    Tensor {
        index: usize,
        dtype: &'static str,
        shape: Vec<u64>,
        value_type: String,
    },
    Record {
        fields: Vec<Field>,
    },
    Table {
        fields: Vec<Field>,
        rows: u64,
    },
    Tuple {
        items: Vec<Schema>,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Selector {
    Field(String),
    Index(usize),
}

type InputPath = (Symbol, Vec<Selector>);

struct Leaf {
    node: NodeId,
    ty: MlirTy,
    elem: ElemKind,
}

pub fn emit_query(
    mut module: Module,
    source: &Module,
    options: &EmitOptions,
    batch: Option<&BatchSpec>,
) -> Result<Export, EmitError> {
    let dtype = options.dtype;
    flatppl_determinizer::is_flatpdl_with_options(&module, &options.lowering_options()).map_err(
        |errors| {
            EmitError::whole(
                errors
                    .into_iter()
                    .map(|e| e.reason)
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        },
    )?;
    let output_node = binding(&module, "outputs")
        .ok_or_else(|| EmitError::whole("declare at least one output"))?;
    let input_nodes = binding(&module, "inputs")
        .map(|id| signature(&module, id))
        .unwrap_or_default();
    let mut names = HashSet::new();
    let mut declared = Vec::new();
    for node in input_nodes {
        let Node::Ref(r) = module.node(node) else {
            return Err(EmitError::at(node, "each input must name a binding"));
        };
        if r.ns != RefNs::SelfMod || !names.insert(r.name) {
            return Err(EmitError::at(
                node,
                "each input must name a distinct local binding",
            ));
        }
        let rhs = module
            .binding(
                module
                    .binding_by_name(r.name)
                    .ok_or_else(|| EmitError::at(node, "input binding is absent"))?,
            )
            .rhs;
        if !builtin(&module, rhs, "elementof") && module.phase_of(rhs) != Some(Phase::Fixed) {
            return Err(EmitError::at(
                rhs,
                "an input must be an elementof leaf or a fixed value",
            ));
        }
        declared.push((module.resolve(r.name).to_string(), node));
    }
    check_query_inputs(&module, &[output_node], &names, &[])?;
    let roots = names;
    let mut arguments = Vec::new();
    let mut inputs = Vec::new();
    for (name, node) in declared {
        let ty = inferred(&module, node)?;
        let value = layout(&mut module, node, ty, &roots, true, dtype, &mut arguments)?;
        inputs.push(Field { name, value });
    }
    // Canonicalization may inline output bindings. Describe their authored names.
    let declared_output = binding(source, "outputs")
        .ok_or_else(|| EmitError::whole("declare at least one output"))?;
    let multiple_outputs = builtin(source, declared_output, "tuple");
    let output_names = signature(source, declared_output)
        .iter()
        .map(|node| match source.node(*node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => Some(source.resolve(r.name).into()),
            _ => None,
        })
        .collect();
    let output_type = inferred(&module, output_node)?;
    let mut results = Vec::new();
    let mut output = layout(
        &mut module,
        output_node,
        output_type,
        &roots,
        false,
        dtype,
        &mut results,
    )?;
    if results.is_empty() {
        return Err(EmitError::at(
            output_node,
            "the output has no tensor leaves",
        ));
    }

    let mut paths = HashMap::new();
    let mut values = HashMap::new();
    let mut args = Vec::new();
    for (index, leaf) in arguments.iter().enumerate() {
        let path = input_path(&module, leaf.node, &roots, &mut paths).ok_or_else(|| {
            EmitError::at(
                leaf.node,
                "cannot map an input leaf to its declared argument",
            )
        })?;
        let value = Value {
            ssa: format!("%arg{index}"),
            ty: leaf.ty.clone(),
            elem: leaf.elem,
        };
        args.push((value.ssa.clone(), value.ty.clone(), value.elem));
        values.insert(path, value);
    }
    let mut emitter = Emitter::with_options(&module, options);
    bind_inputs(&mut emitter, &module, &roots, &mut paths, &values);
    let tensorize = emitter
        .can_tensorize_call(&results.iter().map(|leaf| leaf.node).collect::<Vec<_>>())
        && arguments.iter().all(|leaf| !matches!(leaf.ty, MlirTy::Key));
    let mut returned = Vec::new();
    for leaf in &results {
        let value = emitter.lower_node(leaf.node)?;
        if value.ty != leaf.ty || value.elem != leaf.elem {
            return Err(EmitError::at(
                leaf.node,
                "emitted result type disagrees with the query schema",
            ));
        }
        returned.push(value);
    }
    let returned = returned.iter().collect::<Vec<_>>();
    let stablehlo = if let Some(batch) = batch {
        let primitives = emitter.tensorize_batched("main", &args, &returned, batch);
        // Validate and retain a complete scalar lowering before tensorizing.
        // Unsupported batch/cell combinations use the same scalar program.
        let mapped = emitter.finish_batched("main", &args, &returned, batch)?;
        let tensorized = if primitives.is_some() {
            primitives
        } else if tensorize && !batch.shape.is_empty() {
            let mut e = Emitter::with_options(&module, options);
            let mut physical = Vec::new();
            let mut bound = HashMap::new();
            for (index, leaf) in arguments.iter().enumerate() {
                let path = input_path(&module, leaf.node, &roots, &mut paths).unwrap();
                let (input, internal) =
                    e.batch_input(&values[&path], batch, &batch.input_axes[index])?;
                physical.push((input.ssa, input.ty, input.elem));
                bound.insert(path, internal);
            }
            bind_inputs(&mut e, &module, &roots, &mut paths, &bound);
            results
                .iter()
                .map(|leaf| {
                    let value = e.lower_node(leaf.node)?;
                    let value = e.batch_output(&value, batch);
                    let mut expected = batch.shape.iter().copied().map(Some).collect::<Vec<_>>();
                    match &leaf.ty {
                        MlirTy::Scalar => {}
                        MlirTy::Ranked(dims) => expected.extend(dims),
                        _ => {
                            return Err(EmitError::at(
                                leaf.node,
                                "batched output requires a numeric tensor",
                            ));
                        }
                    }
                    if value.ty != MlirTy::Ranked(expected) || value.elem != leaf.elem {
                        return Err(EmitError::at(
                            leaf.node,
                            "batched result type disagrees with the query schema",
                        ));
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|values| e.finish("main", &physical, &values.iter().collect::<Vec<_>>()))
                .ok()
        } else {
            None
        };
        for field in &mut inputs {
            batch_schema(&mut field.value, batch, true)?;
        }
        batch_schema(&mut output, batch, false)?;
        tensorized.unwrap_or(mapped)
    } else {
        emitter.finish("main", &args, &returned)
    };
    Ok(Export {
        stablehlo,
        entry_point: "main",
        inputs,
        output,
        multiple_outputs,
        output_names,
        batch_shape: batch.map_or_else(Vec::new, |batch| batch.shape.clone()),
    })
}

fn bind_inputs(
    emitter: &mut Emitter<'_>,
    module: &Module,
    roots: &HashSet<Symbol>,
    paths: &mut HashMap<NodeId, Option<InputPath>>,
    values: &HashMap<InputPath, Value>,
) {
    // A declared aggregate has no tensor value. Bind its leaf projections,
    // including aliases and the output projections created by layout().
    for i in 0..module.node_count() {
        let node = NodeId::from_usize(i);
        if let Some(path) = input_path(module, node, roots, paths)
            && let Some(value) = values.get(&path)
        {
            emitter.bind(node, value.clone());
            if let Node::Call(call) = module.node(node)
                && matches!(call.head, CallHead::Builtin(head) if matches!(module.resolve(head), "get" | "get0"))
                && let [container, selector] = call.args.as_ref()
                && let Node::Lit(Scalar::Str(field)) = module.node(*selector)
            {
                emitter.bind_column(*container, field.to_string(), value.clone());
            }
        }
    }
}

fn batch_schema(schema: &mut Schema, batch: &BatchSpec, input: bool) -> Result<(), EmitError> {
    match schema {
        Schema::Tensor { index, shape, .. } => {
            *shape = if input {
                batch.input_shape(shape, &batch.input_axes[*index])?
            } else {
                batch.shape.iter().chain(shape.iter()).copied().collect()
            };
        }
        Schema::Tuple { items } => {
            for item in items {
                batch_schema(item, batch, input)?;
            }
        }
        Schema::Record { fields } | Schema::Table { fields, .. } => {
            for field in fields {
                batch_schema(&mut field.value, batch, input)?;
            }
        }
    }
    Ok(())
}

fn binding(module: &Module, name: &str) -> Option<NodeId> {
    module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .map(|(_, b)| b.rhs)
}

fn builtin(module: &Module, node: NodeId, name: &str) -> bool {
    matches!(module.node(node), Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if module.resolve(s) == name))
}

fn signature(module: &Module, node: NodeId) -> Vec<NodeId> {
    if builtin(module, node, "tuple")
        && let Node::Call(c) = module.node(node)
    {
        c.args.to_vec()
    } else {
        vec![node]
    }
}

fn inferred(module: &Module, node: NodeId) -> Result<Type, EmitError> {
    module
        .type_of(node)
        .cloned()
        .ok_or_else(|| EmitError::at(node, "query value has no inferred type"))
}

fn dimension(node: NodeId, dim: Dim) -> Result<u64, EmitError> {
    match dim {
        Dim::Static(n) => Ok(n.into()),
        Dim::Dynamic => Err(EmitError::at(
            node,
            "the query interface requires a static input and output shape",
        )),
    }
}

fn layout(
    module: &mut Module,
    node: NodeId,
    ty: Type,
    roots: &HashSet<Symbol>,
    input: bool,
    dtype: Dtype,
    leaves: &mut Vec<Leaf>,
) -> Result<Schema, EmitError> {
    match ty {
        Type::Record(fields) => {
            let mut result = Vec::new();
            for (name, ty) in fields.iter() {
                let name = module.resolve(*name).to_string();
                let child = project(
                    module,
                    node,
                    Selector::Field(name.clone()),
                    ty.clone(),
                    roots,
                    input,
                );
                result.push(Field {
                    name,
                    value: layout(module, child, ty.clone(), roots, input, dtype, leaves)?,
                });
            }
            Ok(Schema::Record { fields: result })
        }
        Type::Table { columns, nrows } => {
            let rows = dimension(node, nrows)?;
            let mut fields = Vec::new();
            for (name, ty) in columns.iter() {
                let name = module.resolve(*name).to_string();
                let column = match ty {
                    Type::Record(fields) => Type::Table {
                        columns: fields.clone(),
                        nrows,
                    },
                    ty => Type::Array {
                        shape: Box::new([nrows]),
                        elem: Box::new(ty.clone()),
                    },
                };
                let child = project(
                    module,
                    node,
                    Selector::Field(name.clone()),
                    column.clone(),
                    roots,
                    input,
                );
                fields.push(Field {
                    name,
                    value: layout(module, child, column, roots, input, dtype, leaves)?,
                });
            }
            Ok(Schema::Table { fields, rows })
        }
        Type::Tuple(items) => {
            let mut result = Vec::new();
            for (index, ty) in items.iter().enumerate() {
                let child = project(
                    module,
                    node,
                    Selector::Index(index),
                    ty.clone(),
                    roots,
                    input,
                );
                result.push(layout(
                    module,
                    child,
                    ty.clone(),
                    roots,
                    input,
                    dtype,
                    leaves,
                )?);
            }
            Ok(Schema::Tuple { items: result })
        }
        ty => {
            let (tensor, elem) = mlir_type_of(module, node, dtype)?;
            let shape = match &tensor {
                MlirTy::Scalar => Vec::new(),
                MlirTy::Key => vec![2],
                MlirTy::Ranked(dims) => dims
                    .iter()
                    .map(|d| {
                        d.ok_or_else(|| {
                            EmitError::at(node, "the query interface requires static tensor shapes")
                        })
                    })
                    .collect::<Result<_, _>>()?,
                MlirTy::Tuple(_) => {
                    return Err(EmitError::at(node, "tuple tensors must be flattened"));
                }
            };
            let kind = if tensor == MlirTy::Key {
                "uint64"
            } else {
                match (elem, dtype) {
                    (ElemKind::Bool, _) => "bool",
                    (ElemKind::Int, Dtype::F32) => "int32",
                    (ElemKind::Int, Dtype::F64) => "int64",
                    (ElemKind::Real, Dtype::F32) => "float32",
                    (ElemKind::Real, Dtype::F64) => "float64",
                }
            };
            let result = Schema::Tensor {
                index: leaves.len(),
                dtype: kind,
                shape,
                value_type: module.display_type(&ty),
            };
            leaves.push(Leaf {
                node,
                ty: tensor,
                elem,
            });
            Ok(result)
        }
    }
}

fn project(
    module: &mut Module,
    node: NodeId,
    selector: Selector,
    ty: Type,
    roots: &HashSet<Symbol>,
    input: bool,
) -> NodeId {
    let mut node = node;
    while let Node::Ref(r) = module.node(node)
        && r.ns == RefNs::SelfMod
        && !roots.contains(&r.name)
        && let Some(binding) = module.binding_by_name(r.name)
    {
        node = module.binding(binding).rhs;
    }
    if !input && let Node::Call(c) = module.node(node) {
        match &selector {
            Selector::Field(name)
                if builtin(module, node, "record") || builtin(module, node, "table") =>
            {
                if let Some(field) = c.named.iter().find(|f| module.resolve(f.name) == name) {
                    return field.value;
                }
            }
            Selector::Index(index) if builtin(module, node, "tuple") => {
                if let Some(&value) = c.args.get(*index) {
                    return value;
                }
            }
            _ => {}
        }
    }
    let selector = module.alloc(Node::Lit(match selector {
        Selector::Field(name) => Scalar::Str(name.into()),
        Selector::Index(index) => Scalar::Int(index as i64 + 1),
    }));
    let get = module.intern("get");
    let child = module.alloc(Node::Call(Call {
        head: CallHead::Builtin(get),
        args: Box::new([node, selector]),
        named: Box::new([]),
        inputs: None,
    }));
    module.set_type(child, ty);
    if let Some(phase) = module.phase_of(node) {
        module.set_phase(child, phase);
    }
    child
}

fn input_path(
    module: &Module,
    node: NodeId,
    roots: &HashSet<Symbol>,
    memo: &mut HashMap<NodeId, Option<InputPath>>,
) -> Option<InputPath> {
    if let Some(path) = memo.get(&node) {
        return path.clone();
    }
    let path = match module.node(node) {
        Node::Ref(r) if r.ns == RefNs::SelfMod && roots.contains(&r.name) => {
            Some((r.name, Vec::new()))
        }
        Node::Ref(r) if r.ns == RefNs::SelfMod => module
            .binding_by_name(r.name)
            .and_then(|b| input_path(module, module.binding(b).rhs, roots, memo)),
        Node::Call(c)
            if (builtin(module, node, "get") || builtin(module, node, "get0"))
                && c.args.len() == 2 =>
        {
            let selector = match (module.type_of(c.args[0]), module.node(c.args[1])) {
                (Some(Type::Record(_) | Type::Table { .. }), Node::Lit(Scalar::Str(s))) => {
                    Some(Selector::Field(s.to_string()))
                }
                (Some(Type::Tuple(_)), Node::Lit(Scalar::Int(i))) => {
                    let index = i - i64::from(builtin(module, node, "get"));
                    usize::try_from(index).ok().map(Selector::Index)
                }
                _ => None,
            };
            selector.and_then(|selector| {
                input_path(module, c.args[0], roots, memo).map(|(root, mut path)| {
                    path.push(selector);
                    (root, path)
                })
            })
        }
        _ => None,
    };
    memo.insert(node, path.clone());
    path
}
