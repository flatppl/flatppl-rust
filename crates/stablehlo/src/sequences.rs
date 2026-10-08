//! Callable state is flattened into tensor leaves. A scan emits one while,
//! with a fixed state type and an output buffer for each leaf.

use super::batching::tensor;
use super::*;

#[derive(Clone)]
pub(super) struct Components {
    pub(super) fields: Option<Vec<Symbol>>,
    pub(super) values: Vec<Value>,
}

impl Components {
    pub(super) fn tensor(self, id: NodeId) -> Result<Value, EmitError> {
        if self.fields.is_some() {
            return Err(EmitError::at(id, "record/table requires field projection"));
        }
        Ok(self.values.into_iter().next().expect("one tensor leaf"))
    }

    fn align(mut self, template: &Self, id: NodeId) -> Result<Self, EmitError> {
        if self.fields != template.fields {
            let (Some(fields), Some(expected)) = (&self.fields, &template.fields) else {
                return Err(EmitError::at(id, "scan: step changes the state structure"));
            };
            if fields.len() != expected.len() {
                return Err(EmitError::at(id, "scan: step changes the state fields"));
            }
            self.values = expected
                .iter()
                .map(|name| {
                    fields
                        .iter()
                        .position(|f| f == name)
                        .map(|i| self.values[i].clone())
                        .ok_or_else(|| EmitError::at(id, "scan: step changes the state fields"))
                })
                .collect::<Result<_, _>>()?;
            self.fields = template.fields.clone();
        }
        Ok(self)
    }

    fn with_values(&self, values: &[Value]) -> Self {
        Self {
            fields: self.fields.clone(),
            values: values.to_vec(),
        }
    }
}

impl Emitter<'_> {
    /// Axis substitutions belong to one aggregate application. Shared body
    /// nodes must not retain a previous metric or frame's intermediate values.
    pub(crate) fn lower_with_bindings(
        &mut self,
        body: NodeId,
        bindings: Vec<(NodeId, Value)>,
    ) -> Result<Value, EmitError> {
        let memo = self.memo.clone();
        let columns = self.columns.clone();
        let mut dependent = bindings
            .iter()
            .map(|(id, _)| (*id, true))
            .collect::<HashMap<_, _>>();
        let mut pending = vec![(body, false)];
        while let Some((node, ready)) = pending.pop() {
            if dependent.contains_key(&node) {
                continue;
            }
            let mut children = self.m.node(node).children();
            let resolved = self.resolve_ref_one(node);
            if resolved != node {
                children.push(resolved);
            }
            if ready {
                let changes = children.iter().any(|child| dependent[child]);
                if changes {
                    self.memo.remove(&node);
                }
                dependent.insert(node, changes);
            } else {
                pending.push((node, true));
                pending.extend(children.into_iter().map(|child| (child, false)));
            }
        }
        for (id, value) in bindings {
            self.bind(id, value);
        }
        let result = self.lower_node(body);
        self.memo = memo;
        self.columns = columns;
        result
    }

    pub(super) fn sequence_item_type(&self, id: NodeId) -> Result<Type, EmitError> {
        match self.type_of(id) {
            Some(Type::Array { elem, .. } | Type::TVector { elem, .. }) => Ok((**elem).clone()),
            Some(Type::Table { columns, .. }) => Ok(Type::Record(columns.clone())),
            Some(ty) => Ok(ty.clone()),
            None => Err(EmitError::at(id, "sequence has no inferred type")),
        }
    }

    pub(super) fn callable_parts(
        &self,
        id: NodeId,
    ) -> Result<(NodeId, Vec<(Symbol, Ref)>), EmitError> {
        let function = self
            .resolve_functionof(id)
            .ok_or_else(|| EmitError::at(id, "expected a reified deterministic function"))?;
        let Node::Call(c) = self.m.node(function) else {
            unreachable!()
        };
        let body = *c
            .args
            .first()
            .ok_or_else(|| EmitError::at(id, "function has no body"))?;
        let entries = match &c.inputs {
            Some(Inputs::Spec(entries)) => entries.to_vec(),
            Some(Inputs::Auto) => self
                .m
                .auto_inputs_of(function)
                .ok_or_else(|| EmitError::at(id, "function has unresolved automatic inputs"))?
                .to_vec(),
            None => unreachable!(),
        };
        Ok((body, entries))
    }

    pub(super) fn lower_components(&mut self, id: NodeId) -> Result<Components, EmitError> {
        if let Some(value) = self.memo.get(&id).cloned() {
            return Ok(Components {
                fields: None,
                values: vec![self.typed_axes(id, value)],
            });
        }
        let mut resolved = id;
        for _ in 0..64 {
            let next = self.resolve_ref_one(resolved);
            if next == resolved {
                break;
            }
            resolved = next;
        }
        let fields = match self.type_of(id) {
            Some(
                Type::Record(fields)
                | Type::Table {
                    columns: fields, ..
                },
            ) => Some(fields.iter().map(|(name, _)| *name).collect::<Vec<_>>()),
            _ => None,
        };
        if let Some(fields) = &fields {
            let values = fields
                .iter()
                .map(|&name| {
                    self.columns
                        .get(&(resolved, self.m.resolve(name).to_owned()))
                        .cloned()
                })
                .collect::<Option<Vec<_>>>();
            if let Some(values) = values {
                return Ok(Components {
                    fields: Some(fields.clone()),
                    values,
                });
            }
        }
        if let Node::Call(c) = self.m.node(resolved)
            && let CallHead::Builtin(head) = c.head
        {
            match self.m.resolve(head) {
                "scan" => return self.lower_scan(resolved, &c.args),
                "record" | "table" => {
                    let mut fields = Vec::with_capacity(c.named.len());
                    let mut values = Vec::with_capacity(c.named.len());
                    for field in &c.named {
                        fields.push(field.name);
                        let value = self.lower_node(field.value)?;
                        values.push(self.typed_axes(field.value, value));
                    }
                    return Ok(Components {
                        fields: Some(fields),
                        values,
                    });
                }
                _ => (),
            }
        }
        if fields.is_some() {
            return Err(EmitError::at(
                id,
                "record/table producer has no field lowering",
            ));
        }
        let value = self.lower_node(id)?;
        Ok(Components {
            fields: None,
            values: vec![self.typed_axes(id, value)],
        })
    }

    pub(super) fn call_components(
        &mut self,
        body: NodeId,
        entries: &[(Symbol, Ref)],
        args: &[Components],
        types: &[Type],
    ) -> Result<Components, EmitError> {
        // Region-local values must not escape through the node/column caches.
        // Captured query arguments retain their outer bindings.
        let memo = self.memo.clone();
        let columns = self.columns.clone();
        let inputs = self.scoped_inputs.clone();
        let mut dependent = HashMap::new();
        let mut stack = vec![(body, false)];
        while let Some((node, ready)) = stack.pop() {
            if dependent.contains_key(&node) {
                continue;
            }
            let mut children = self.m.node(node).children();
            if let Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) = self.m.node(node)
                && let Some(binding) = self.m.binding_by_name(*name)
            {
                children.push(self.m.binding(binding).rhs);
            }
            if let Node::Ref(reference) = self.m.node(node)
                && let Some(index) = entries.iter().position(|(_, r)| r == reference)
            {
                dependent.insert(node, true);
                self.scoped_inputs.insert(node, types[index].clone());
                self.memo.remove(&node);
                let arg = &args[index];
                if let Some(fields) = &arg.fields {
                    for (&field, value) in fields.iter().zip(&arg.values) {
                        self.bind_column(node, self.m.resolve(field).to_owned(), value.clone());
                    }
                } else {
                    self.bind(node, arg.values[0].clone());
                }
            } else if ready {
                let changes = children.iter().any(|child| dependent[child]);
                if changes {
                    self.memo.remove(&node);
                }
                dependent.insert(node, changes);
            } else {
                stack.push((node, true));
                stack.extend(children.into_iter().map(|child| (child, false)));
            }
        }
        let module = self.inference_module.get_or_insert_with(|| self.m.clone());
        let seeds = self
            .scoped_inputs
            .iter()
            .map(|(&id, ty)| (id, ty.clone()))
            .collect::<Vec<_>>();
        let (types, diagnostics) = flatppl_infer::infer_expression(module, body, &seeds);
        let previous_types = std::mem::replace(&mut self.scoped_types, types);
        let result = if let Some(error) = diagnostics
            .iter()
            .find(|d| d.severity == flatppl_infer::Severity::Error)
        {
            Err(EmitError::at(
                error.node.unwrap_or(body),
                error.message.clone(),
            ))
        } else if let Node::Ref(reference) = self.m.node(body)
            && let Some(index) = entries.iter().position(|(_, r)| r == reference)
        {
            Ok(args[index].clone())
        } else {
            self.lower_components(body)
        };
        self.memo = memo;
        self.columns = columns;
        self.scoped_inputs = inputs;
        self.scoped_types = previous_types;
        result
    }

    pub(super) fn sequence_field(&mut self, args: &[NodeId]) -> Option<Result<Value, EmitError>> {
        let [container, selector] = <[NodeId; 2]>::try_from(args).ok()?;
        let field = match self.m.node(selector) {
            Node::Lit(Scalar::Str(s)) => s.as_ref(),
            Node::Const(name) => self.m.resolve(*name),
            _ => return None,
        };
        if !matches!(
            self.type_of(container),
            Some(Type::Table { .. } | Type::Record(_))
        ) {
            return None;
        }
        Some(self.lower_components(container).and_then(|parts| {
            let index = parts
                .fields
                .as_ref()
                .and_then(|fields| {
                    fields
                        .iter()
                        .position(|&name| self.m.resolve(name) == field)
                })
                .ok_or_else(|| EmitError::at(container, "unknown record/table field"))?;
            Ok(parts.values[index].clone())
        }))
    }

    pub(super) fn lower_scan(
        &mut self,
        id: NodeId,
        args: &[NodeId],
    ) -> Result<Components, EmitError> {
        let [function, init, xs] = crate::ops::args_exact(id, args)?;
        let item_type = self.sequence_item_type(xs)?;
        let state_type = self.sequence_item_type(id)?;
        if self.broadcast_frame != self.call_frame {
            return Err(EmitError::at(id, "scan inside broadcast has no lowering"));
        }
        let (body, entries) = self.callable_parts(function)?;
        if entries.len() != 2 {
            return Err(EmitError::at(id, "scan: step must take state and input"));
        }
        let mut state = self.lower_components(init)?;
        let xs = self.lower_components(xs)?;
        let Some(Some(n)) = xs
            .values
            .first()
            .and_then(|v| shape(&v.ty).get(self.batch_rank(v)))
            .copied()
        else {
            return Err(EmitError::at(
                id,
                "scan: input requires a static leading extent",
            ));
        };
        if xs.values.iter().any(|v| {
            shape(&v.ty).get(self.batch_rank(v)) != Some(&Some(n))
                || shape(&v.ty).iter().any(Option::is_none)
                || self.axes_of(v).layers.first() != Some(&1)
        }) {
            return Err(EmitError::at(
                id,
                "scan: expected a vector or equal-length table columns",
            ));
        }
        if let Some(fields) = &state.fields {
            let cached = fields
                .iter()
                .map(|&field| {
                    self.columns
                        .get(&(id, self.m.resolve(field).to_owned()))
                        .cloned()
                })
                .collect::<Option<Vec<_>>>();
            if let Some(values) = cached {
                return Ok(state.with_values(&values));
            }
        } else if let Some(value) = self.memo.get(&id) {
            return Ok(state.with_values(std::slice::from_ref(value)));
        }

        let carry_types = match (self.type_of(id), &state.fields) {
            (Some(Type::Table { columns, .. }), Some(fields)) => fields
                .iter()
                .map(|name| {
                    columns
                        .iter()
                        .find(|(field, _)| field == name)
                        .map(|(_, ty)| ty.clone())
                        .ok_or_else(|| EmitError::at(id, "scan: unresolved state field"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            (Some(Type::Array { elem, .. }), None) => vec![(**elem).clone()],
            _ => return Err(EmitError::at(id, "scan: unresolved state type")),
        };
        for (value, ty) in state.values.iter_mut().zip(carry_types) {
            let (shape, kind) = crate::types::mlir_type_of_ty(id, &ty, self.dtype)?;
            if self.cell_ty(value) != shape || elem_rank(value.elem) > elem_rank(kind) {
                return Err(EmitError::at(
                    id,
                    "scan: initial value does not fit the state type",
                ));
            }
            *value = self.convert(value, kind);
            // A captured mapped argument may enter the state at any step.
            // Fix the full host frame before creating the loop carry types.
            let frame = self.call_frame.clone();
            *value = self.finish_broadcast(value, &frame, frame.len());
        }
        let mut buffers = Vec::with_capacity(state.values.len());
        for value in &state.values {
            let mut dims = shape(&value.ty).to_vec();
            dims.insert(self.batch_rank(value), Some(n));
            let zero = match value.elem {
                ElemKind::Real => self.scalar(0.0),
                ElemKind::Int => self.int_value_const(0),
                ElemKind::Bool => self.bool_value_const(false),
            };
            let buffer = self.broadcast_in_dim(&zero, &[], tensor(dims));
            let mut axes = self.axes_of(value);
            axes.layers.insert(0, 1);
            buffers.push(self.axes_view(&buffer, axes));
        }
        if n > 0 {
            let counter = self.int_value_const(0);
            let mut inits = vec![counter];
            inits.extend(xs.values.clone());
            let start = inits.len();
            inits.extend(state.values.clone());
            let out_start = inits.len();
            inits.extend(buffers);
            let types = inits
                .iter()
                .map(|v| v.ty.render(self.dtype, v.elem))
                .collect::<Vec<_>>();
            let results = self.try_while_loop(
                &inits,
                &types,
                |e, args| {
                    let limit = e.int_value_const(n as i64);
                    e.compare("LT", &args[0], &limit)
                },
                |e, args| {
                    let mut items = Vec::with_capacity(xs.values.len());
                    for (value, original) in args[1..start].iter().zip(&xs.values) {
                        let value = e.axes_view(value, e.axes_of(original));
                        items.push(e.scan_slice(&value, &args[0]));
                    }
                    let mut carries = Vec::with_capacity(state.values.len());
                    for (value, original) in args[start..out_start].iter().zip(&state.values) {
                        carries.push(e.axes_view(value, e.axes_of(original)));
                    }
                    let output = e
                        .call_components(
                            body,
                            &entries,
                            &[state.with_values(&carries), xs.with_values(&items)],
                            &[state_type.clone(), item_type.clone()],
                        )?
                        .align(&state, id)?;
                    let mut next = Vec::with_capacity(args.len());
                    let one = e.int_value_const(1);
                    next.push(e.add(&args[0], &one));
                    next.extend_from_slice(&args[1..start]);
                    for (value, expected) in output.values.iter().zip(&state.values) {
                        let frame = e.call_frame.clone();
                        let value = e.finish_broadcast(value, &frame, frame.len());
                        if value.ty != expected.ty
                            || elem_rank(value.elem) > elem_rank(expected.elem)
                        {
                            return Err(EmitError::at(id, "scan: step changes the state type"));
                        }
                        next.push(e.convert(&value, expected.elem));
                    }
                    let updates = args[out_start..]
                        .iter()
                        .zip(&next[start..out_start])
                        .map(|(buffer, value)| e.scan_update(buffer, value, &args[0]))
                        .collect::<Vec<_>>();
                    next.extend(updates);
                    Ok(next)
                },
            )?;
            buffers = results[out_start..].to_vec();
        }
        for (buffer, value) in buffers.iter_mut().zip(&state.values) {
            let mut axes = self.axes_of(value);
            axes.layers.insert(0, 1);
            *buffer = self.axes_view(buffer, axes);
        }
        if let Some(fields) = &state.fields {
            for (&field, value) in fields.iter().zip(&buffers) {
                self.bind_column(id, self.m.resolve(field).to_owned(), value.clone());
            }
        } else {
            self.bind(id, buffers[0].clone());
        }
        Ok(state.with_values(&buffers))
    }

    pub(super) fn scan_slice(&mut self, input: &Value, index: &Value) -> Value {
        let mut dims = shape(&input.ty).to_vec();
        let axis = self.batch_rank(input);
        dims[axis] = Some(1);
        let sliced_ty = tensor(dims.clone());
        let zero = self.int_value_const(0);
        let mut starts = vec![zero.ssa.as_str(); dims.len()];
        starts[axis] = &index.ssa;
        let starts = starts.join(", ");
        let indices = vec![index.ty.render(self.dtype, index.elem); dims.len()].join(", ");
        let sizes = dims
            .iter()
            .map(|d| d.unwrap().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let from = input.ty.render(self.dtype, input.elem);
        let to = sliced_ty.render(self.dtype, input.elem);
        let ssa = self.pure(format!(
            "stablehlo.dynamic_slice {}, {starts}, sizes = [{sizes}] : ({from}, {indices}) -> {to}",
            input.ssa
        ));
        dims.remove(axis);
        let value = self.reshape(
            &Value {
                ssa,
                ty: sliced_ty,
                elem: input.elem,
            },
            tensor(dims),
        );
        let mut axes = self.axes_of(input);
        axes.layers.remove(0);
        self.axes_view(&value, axes)
    }

    pub(super) fn scan_update(&mut self, buffer: &Value, value: &Value, index: &Value) -> Value {
        let axis = self.batch_rank(value);
        let mut dims = shape(&value.ty).to_vec();
        dims.insert(axis, Some(1));
        let update = self.reshape(value, tensor(dims));
        let zero = self.int_value_const(0);
        let rank = shape(&buffer.ty).len();
        let mut starts = vec![zero.ssa.as_str(); rank];
        starts[axis] = &index.ssa;
        let starts = starts.join(", ");
        let indices = vec![index.ty.render(self.dtype, index.elem); rank].join(", ");
        let ty = buffer.ty.render(self.dtype, buffer.elem);
        let update_ty = update.ty.render(self.dtype, update.elem);
        let ssa = self.pure(format!("stablehlo.dynamic_update_slice {}, {}, {starts} : ({ty}, {update_ty}, {indices}) -> {ty}", buffer.ssa, update.ssa));
        Value {
            ssa,
            ..buffer.clone()
        }
    }
}
