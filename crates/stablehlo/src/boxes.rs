//! Black-box queries as portable `stablehlo.custom_call @flatppl.box` ops.
//!
//! Each box query in FlatPDL becomes one custom call. Its attributes carry the
//! box, the member, the operation and the query term. The executor binds the
//! call, for example to an HTTP client of the flatppl-bridge protocol. The
//! contract is in `docs/box-custom-call.md`.

use std::fmt::Write as _;
use std::sync::Arc;

use flatppl_core::{BoxBinding, BoxKind, CallHead, Module, Node, NodeId, Scalar, Type};

use super::Emitter;
use crate::mlir::{ElemKind, Lowered, MlirTy, Value};
use crate::refuse::EmitError;

/// The custom-call target of every box query.
pub const BOX_TARGET: &str = "flatppl.box";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Apply,
    LogDensity,
    Sample,
}

impl Operation {
    fn name(self) -> &'static str {
        match self {
            Operation::Apply => "apply",
            Operation::LogDensity => "logdensity",
            Operation::Sample => "sample",
        }
    }
}

struct Target {
    binding: Arc<BoxBinding>,
    member: String,
}

/// The box member `node` names, through bare-name aliases.
fn target(m: &Module, node: NodeId) -> Option<Target> {
    let (binding, member) = flatppl_determinizer::box_member(m, node)?;
    Some(Target { binding, member })
}

/// One argument of a query: its wire name and value node. `constant` holds the
/// argument fixed across the rows of a broadcast.
struct Argument {
    name: &'static str,
    node: NodeId,
    constant: bool,
}

impl<'m> Emitter<'m> {
    /// Lower `builtin_logdensityof` or `builtin_sample` on a box member, or
    /// `None` when the tag is no box member.
    pub(super) fn lower_box_primitive(
        &mut self,
        id: NodeId,
        name: &str,
        args: &[NodeId],
    ) -> Option<Result<Value, EmitError>> {
        let tag = usize::from(name == "builtin_sample");
        let found = target(self.m, *args.get(tag)?)?;
        let kind = found.binding.members[found.member.as_str()].kind;
        let inputs = usize::from(kind == BoxKind::Kernel);
        Some(match name {
            "builtin_logdensityof" => {
                if args.len() != 2 + inputs {
                    return Some(Err(EmitError::at(id, "builtin_logdensityof: wrong arity")));
                }
                let mut arguments = Vec::new();
                if inputs == 1 {
                    arguments.push(Argument {
                        name: "input",
                        node: args[1],
                        constant: true,
                    });
                }
                arguments.push(Argument {
                    name: "x",
                    node: args[1 + inputs],
                    constant: true,
                });
                self.emit_box_call(
                    id,
                    &found,
                    Operation::LogDensity,
                    arguments,
                    None,
                    &[],
                    false,
                )
            }
            "builtin_sample" => {
                let rest = &args[2..];
                if rest.len() < inputs {
                    return Some(Err(EmitError::at(
                        id,
                        "builtin_sample: missing kernel input",
                    )));
                }
                let arguments = rest[..inputs]
                    .iter()
                    .map(|&node| Argument {
                        name: "input",
                        node,
                        constant: true,
                    })
                    .collect();
                let sizes = rest[inputs..]
                    .iter()
                    .map(|&dim| match self.m.node(dim) {
                        Node::Lit(Scalar::Int(n)) if *n > 0 => Ok(*n as u64),
                        _ => Err(EmitError::at(
                            dim,
                            "box sample sizes must be positive integer literals",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>();
                match sizes {
                    Ok(sizes) => self.emit_box_call(
                        id,
                        &found,
                        Operation::Sample,
                        arguments,
                        Some(args[0]),
                        &sizes,
                        false,
                    ),
                    Err(error) => Err(error),
                }
            }
            _ => return None,
        })
    }

    /// Lower an application `B.f(input)` of a box function.
    pub(super) fn lower_box_application(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
    ) -> Option<Result<Value, EmitError>> {
        let found = target(self.m, callee)?;
        let named = match self.m.node(id) {
            Node::Call(c) => !c.named.is_empty(),
            _ => false,
        };
        let [input] = args else {
            return Some(Err(EmitError::at(
                id,
                "a box application takes one input; determinize binds its arguments",
            )));
        };
        if named {
            return Some(Err(EmitError::at(
                id,
                "box arguments must be bound by determinize",
            )));
        }
        let arguments = vec![Argument {
            name: "input",
            node: *input,
            constant: true,
        }];
        Some(self.emit_box_call(id, &found, Operation::Apply, arguments, None, &[], false))
    }

    /// Lower `broadcast(builtin_logdensityof, B.K, ...)` or `broadcast(B.f, ...)`
    /// as one broadcast query.
    pub(super) fn lower_box_broadcast(
        &mut self,
        id: NodeId,
        args: &[NodeId],
    ) -> Option<Result<Value, EmitError>> {
        let callee = *args.first()?;
        let is_density = matches!(self.m.node(callee), Node::Const(s) if self.m.resolve(*s) == "builtin_logdensityof");
        if is_density {
            let found = target(self.m, *args.get(1)?)?;
            let kind = found.binding.members[found.member.as_str()].kind;
            let names: &[&'static str] = if kind == BoxKind::Kernel {
                &["input", "x"]
            } else {
                &["x"]
            };
            if args.len() != 2 + names.len() {
                return Some(Err(EmitError::at(id, "box density broadcast: wrong arity")));
            }
            let arguments = names
                .iter()
                .zip(&args[2..])
                .map(|(&name, &node)| self.broadcast_argument(name, node))
                .collect();
            return Some(self.emit_box_call(
                id,
                &found,
                Operation::LogDensity,
                arguments,
                None,
                &[],
                true,
            ));
        }
        let found = target(self.m, callee)?;
        let [input] = &args[1..] else {
            return Some(Err(EmitError::at(
                id,
                "a box function broadcast takes one input",
            )));
        };
        let argument = self.broadcast_argument("input", *input);
        Some(self.emit_box_call(
            id,
            &found,
            Operation::Apply,
            vec![argument],
            None,
            &[],
            true,
        ))
    }

    /// A broadcast operand: a size-one array of a value holds that value
    /// constant, and any other array iterates over its elements.
    fn broadcast_argument(&self, name: &'static str, node: NodeId) -> Argument {
        if let Some(inner) = self.singleton(node) {
            return Argument {
                name,
                node: inner,
                constant: true,
            };
        }
        let constant = !matches!(
            self.type_of(node),
            Some(Type::Array { .. } | Type::Table { .. })
        );
        Argument {
            name,
            node,
            constant,
        }
    }

    /// The value inside `vector(value)`, nested any number of times, or inside
    /// `broadcast(record, f = vector(v), ...)`.
    fn singleton(&self, node: NodeId) -> Option<NodeId> {
        let node = self.resolve_ref_one(node);
        let Node::Call(c) = self.m.node(node) else {
            return None;
        };
        let CallHead::Builtin(head) = c.head else {
            return None;
        };
        match (self.m.resolve(head), &c.args[..]) {
            ("vector", [inner]) => Some(self.singleton(*inner).unwrap_or(*inner)),
            ("broadcast", [record])
                if matches!(self.m.node(*record), Node::Const(s) if self.m.resolve(*s) == "record")
                    && !c.named.is_empty() =>
            {
                let fields: Option<Vec<_>> = c
                    .named
                    .iter()
                    .map(|n| self.singleton(n.value).map(|v| (n.name, v)))
                    .collect();
                // The emitter allocates no nodes, so the record stays this node.
                // `box_operand` lowers its unwrapped fields.
                fields.map(|_| node)
            }
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_box_call(
        &mut self,
        id: NodeId,
        found: &Target,
        operation: Operation,
        arguments: Vec<Argument>,
        rng: Option<NodeId>,
        sizes: &[u64],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        let box_name = &found.binding.name;
        let member = &found.member;
        if self.region_depth > 0 || !self.broadcast_frame.is_empty() || !self.call_frame.is_empty()
        {
            return Err(EmitError::at(
                id,
                format!(
                    "box query `{box_name}.{member}` sits inside a loop or a broadcast body; \
                     a box query must be a top-level operation (broadcast the member directly)"
                ),
            ));
        }
        if self.host_batch || self.runtime_batch_rank > 0 {
            return Err(EmitError::at(
                id,
                "box queries batch in the executor, not through host batch axes",
            ));
        }
        if operation == Operation::Sample && self.restrict_enzyme_compatible() {
            return Err(EmitError::at(
                id,
                "Enzyme restrictions require a deterministic query",
            ));
        }

        let mut operands: Vec<Value> = Vec::new();
        let mut rows: Option<Vec<u64>> = None;
        let mut attributes = Vec::new();
        if let Some(rng) = rng {
            let key = self.lower_node(rng)?;
            if key.ty != MlirTy::Key {
                return Err(EmitError::at(rng, "box sample needs an RNG state"));
            }
            self.set_cur_key(key.clone());
            operands.push(key);
        }
        for argument in &arguments {
            let (element, outer) = self.box_argument_type(argument)?;
            if let Some(outer) = &outer {
                match &rows {
                    None => rows = Some(outer.clone()),
                    Some(rows) if rows == outer => {}
                    Some(_) => {
                        return Err(EmitError::at(
                            argument.node,
                            "box broadcast operands need one common shape",
                        ));
                    }
                }
            }
            let valueset = set_text(self.m, &element).ok_or_else(|| {
                EmitError::at(
                    argument.node,
                    format!("box argument `{}` has no value-set form", argument.name),
                )
            })?;
            let lowered = self.box_operand(argument)?;
            let fields = match &lowered {
                Lowered::Tensor(_) => None,
                Lowered::Record(fields) => {
                    if fields.iter().any(|(_, f)| matches!(f, Lowered::Record(_))) {
                        return Err(EmitError::at(
                            argument.node,
                            "box arguments with nested records are not supported",
                        ));
                    }
                    Some(fields.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>())
                }
            };
            let mut leaves = Vec::new();
            lowered.leaves(&mut leaves);
            operands.extend(leaves);
            let mut attribute = format!(
                "{{name = {}, valueset = {}, rowwise = {}",
                quote(argument.name),
                quote(&valueset),
                outer.is_some()
            );
            if let Some(fields) = fields {
                let names: Vec<_> = fields.iter().map(|f| quote(f)).collect();
                let _ = write!(attribute, ", fields = [{}]", names.join(", "));
            }
            attribute.push('}');
            attributes.push(attribute);
        }
        let rows = rows.unwrap_or_default();

        let (cell, cell_kind) = self.box_result_type(id, operation, broadcast)?;
        let mut result_dims: Vec<Option<u64>> = rows.iter().map(|&n| Some(n)).collect();
        result_dims.extend(sizes.iter().map(|&n| Some(n)));
        match &cell {
            MlirTy::Scalar => {}
            MlirTy::Ranked(dims) => result_dims.extend(dims.iter().copied()),
            _ => return Err(EmitError::at(id, "box results are tensors")),
        }
        if result_dims.contains(&None) {
            return Err(EmitError::at(id, "box results need static shapes"));
        }
        let result_ty = if result_dims.is_empty() {
            MlirTy::Scalar
        } else {
            MlirTy::Ranked(result_dims)
        };

        let callee = if operation == Operation::LogDensity {
            "builtin_logdensityof"
        } else if operation == Operation::Sample {
            "builtin_sample"
        } else {
            ""
        };
        let names: Vec<&str> = arguments.iter().map(|a| a.name).collect();
        let query = query_term(
            callee,
            member,
            &names,
            rng.is_some(),
            sizes,
            !rows.is_empty(),
        );
        let size_list: Vec<String> = sizes.iter().map(u64::to_string).collect();
        let row_list: Vec<String> = rows.iter().map(u64::to_string).collect();
        let config = format!(
            "{{box = {}, member = {}, operation = {}, query = {}, arguments = [{}], rows = [{}], sizes = [{}]}}",
            quote(box_name),
            quote(member),
            quote(operation.name()),
            quote(&query),
            attributes.join(", "),
            row_list.join(", "),
            size_list.join(", "),
        );
        let operand_list: Vec<&str> = operands.iter().map(|v| v.ssa.as_str()).collect();
        let operand_types: Vec<String> = operands
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect();
        let result_text = result_ty.render(self.dtype, cell_kind);
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.custom_call @{BOX_TARGET}({}) {{api_version = 4 : i32, backend_config = {config}}} : ({}) -> {result_text}",
            operand_list.join(", "),
            operand_types.join(", "),
        ));
        let value = Value {
            ssa,
            ty: result_ty,
            elem: cell_kind,
        };
        if operation == Operation::Sample {
            // The box draws from its own RNG, seeded from this state. The model's
            // stream advances by one draw so later samples stay independent.
            let key = self.cur_key();
            let (state, bits) = (self.fresh(), self.fresh());
            self.push(&format!(
                "{state}, {bits} = stablehlo.rng_bit_generator {}, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)",
                key.ssa
            ));
            let advanced = Value {
                ssa: state,
                ty: MlirTy::Key,
                elem: ElemKind::Real,
            };
            self.record_sample_key(id, advanced.clone());
            self.set_cur_key(advanced);
        }
        Ok(value)
    }

    /// The element type of an argument and, for a row-wise argument, its row
    /// shape.
    fn box_argument_type(
        &self,
        argument: &Argument,
    ) -> Result<(Type, Option<Vec<u64>>), EmitError> {
        let node = argument.node;
        let ty = self.box_value_type(node)?;
        if argument.constant {
            return Ok((ty, None));
        }
        let static_dims = |dims: &[flatppl_core::Dim]| -> Result<Vec<u64>, EmitError> {
            dims.iter()
                .map(|d| match d {
                    flatppl_core::Dim::Static(n) => Ok(u64::from(*n)),
                    _ => Err(EmitError::at(
                        node,
                        "box broadcast operands need static shapes",
                    )),
                })
                .collect()
        };
        match ty {
            Type::Array { shape, elem } => Ok((*elem, Some(static_dims(&shape)?))),
            Type::Table { columns, nrows } => Ok((
                Type::Record(columns),
                Some(static_dims(std::slice::from_ref(&nrows))?),
            )),
            other => Ok((other, None)),
        }
    }

    /// The FlatPPL type of a box operand; a held-constant broadcast record has
    /// the type of one of its rows.
    fn box_value_type(&self, node: NodeId) -> Result<Type, EmitError> {
        if let Some(fields) = self.singleton_record_fields(node) {
            let fields = fields
                .into_iter()
                .map(|(name, value)| {
                    self.type_of(value)
                        .cloned()
                        .map(|ty| (name, ty))
                        .ok_or_else(|| EmitError::at(value, "node has no inferred type"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Type::Record(fields.into()));
        }
        self.type_of(node)
            .cloned()
            .ok_or_else(|| EmitError::at(node, "node has no inferred type"))
    }

    fn singleton_record_fields(&self, node: NodeId) -> Option<Vec<(flatppl_core::Symbol, NodeId)>> {
        let node = self.resolve_ref_one(node);
        let Node::Call(c) = self.m.node(node) else {
            return None;
        };
        let CallHead::Builtin(head) = c.head else {
            return None;
        };
        if self.m.resolve(head) != "broadcast" || c.named.is_empty() {
            return None;
        }
        c.named
            .iter()
            .map(|n| self.singleton(n.value).map(|v| (n.name, v)))
            .collect()
    }

    /// Lower an argument to its tensors. A held-constant broadcast record
    /// lowers field by field.
    fn box_operand(&mut self, argument: &Argument) -> Result<Lowered, EmitError> {
        if let Some(fields) = self.singleton_record_fields(argument.node) {
            let mut lowered = Vec::with_capacity(fields.len());
            for (name, value) in fields {
                let name = self.m.resolve(name).to_string();
                lowered.push((name, self.lower_value(value)?));
            }
            return Ok(Lowered::Record(lowered));
        }
        self.lower_value(argument.node)
    }

    /// The tensor type of one row of the result.
    fn box_result_type(
        &self,
        id: NodeId,
        operation: Operation,
        broadcast: bool,
    ) -> Result<(MlirTy, ElemKind), EmitError> {
        match operation {
            Operation::LogDensity => Ok((MlirTy::Scalar, ElemKind::Real)),
            Operation::Sample => {
                let Some(Type::Tuple(parts)) = self.type_of(id) else {
                    return Err(EmitError::at(id, "box sample has no inferred variate"));
                };
                // The inferred value already includes the sample sizes.
                let variate = match &parts[0] {
                    Type::Array { elem, .. } if self.sample_sized(id) => elem.as_ref().clone(),
                    other => other.clone(),
                };
                crate::types::mlir_type_of_ty(id, &variate, self.dtype)
            }
            Operation::Apply => {
                let ty = self
                    .type_of(id)
                    .ok_or_else(|| EmitError::at(id, "box application has no inferred output"))?;
                let cell = match ty {
                    Type::Array { elem, .. } if broadcast => elem.as_ref(),
                    other => other,
                };
                if matches!(cell, Type::Record(_) | Type::Table { .. } | Type::Any) {
                    return Err(EmitError::at(
                        id,
                        "a box function needs a declared array or scalar output",
                    ));
                }
                crate::types::mlir_type_of_ty(id, cell, self.dtype)
            }
        }
    }

    fn sample_sized(&self, id: NodeId) -> bool {
        let Node::Call(c) = self.m.node(id) else {
            return false;
        };
        c.args
            .last()
            .is_some_and(|&last| matches!(self.m.node(last), Node::Lit(Scalar::Int(_))))
    }
}

/// The protocol query term of one row, with the arguments as bindings.
fn query_term(
    callee: &str,
    member: &str,
    names: &[&str],
    rng: bool,
    sizes: &[u64],
    broadcast: bool,
) -> String {
    let mut args: Vec<String> = Vec::new();
    if rng {
        args.push("rng".into());
    }
    args.push(format!("B.{member}"));
    args.extend(names.iter().map(|n| n.to_string()));
    args.extend(sizes.iter().map(u64::to_string));
    match (callee, broadcast) {
        ("", false) => format!("B.{member}({})", names.join(", ")),
        ("", true) => format!("broadcast(B.{member}, {})", names.join(", ")),
        (callee, false) => format!("{callee}({})", args.join(", ")),
        (callee, true) => format!("broadcast({callee}, {})", args.join(", ")),
    }
}

/// The FlatPPL set whose elements have type `ty`.
pub(crate) fn set_text(m: &Module, ty: &Type) -> Option<String> {
    use flatppl_core::{Dim, ScalarType};
    Some(match ty {
        Type::Scalar(ScalarType::Real) => "reals".into(),
        Type::Scalar(ScalarType::Integer) => "integers".into(),
        Type::Scalar(ScalarType::Boolean) => "booleans".into(),
        Type::Array { shape, elem } => {
            let dims = shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => Some(n.to_string()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let inner = set_text(m, elem)?;
            match &dims[..] {
                [n] => format!("cartpow({inner}, {n})"),
                dims => format!("cartpow({inner}, [{}])", dims.join(", ")),
            }
        }
        Type::Record(fields) => {
            let fields = fields
                .iter()
                .map(|(name, ty)| Some(format!("{} = {}", m.resolve(*name), set_text(m, ty)?)))
                .collect::<Option<Vec<_>>>()?;
            format!("cartprod({})", fields.join(", "))
        }
        _ => return None,
    })
}

fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\22"),
            '\\' => out.push_str("\\5C"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
