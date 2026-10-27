//! Callable state is flattened into tensor leaves. A scan emits one while,
//! with a fixed state type and an output buffer for each leaf.

use super::batching::tensor;
use super::*;

#[derive(Clone)]
pub(super) struct Components {
    pub(super) fields: Option<Vec<Symbol>>,
    /// The tensor leaves, depth-first in declared field order.
    pub(super) values: Vec<Value>,
    /// One structure per record field when some field is itself a record;
    /// `None` when every field is one leaf. Its tensors are placeholders: the
    /// leaves always come from `values`.
    layout: Option<Vec<Lowered>>,
}

impl Components {
    pub(super) fn flat(fields: Option<Vec<Symbol>>, values: Vec<Value>) -> Self {
        Self {
            fields,
            values,
            layout: None,
        }
    }

    /// A record whose fields may themselves be records.
    pub(super) fn record(fields: Vec<Symbol>, parts: Vec<Lowered>) -> Self {
        let mut values = Vec::new();
        for part in parts.iter().cloned() {
            part.leaves(&mut values);
        }
        let nested = parts.iter().any(|part| matches!(part, Lowered::Record(_)));
        Self {
            fields: Some(fields),
            values,
            layout: nested.then_some(parts),
        }
    }

    /// One value per record field, rebuilt from the leaves.
    pub(super) fn field_parts(&self) -> Vec<Lowered> {
        let Some(layout) = &self.layout else {
            return self.values.iter().cloned().map(Lowered::Tensor).collect();
        };
        let mut leaves = self.values.iter().cloned();
        layout
            .iter()
            .map(|part| part.with_leaves(&mut leaves))
            .collect()
    }

    pub(super) fn tensor(self, id: NodeId) -> Result<Value, EmitError> {
        if self.fields.is_some() || self.values.len() != 1 {
            return Err(EmitError::at(id, "structured object requires a projection"));
        }
        Ok(self.values.into_iter().next().expect("one tensor leaf"))
    }

    /// Whether both carry the same fields with the same nesting.
    fn same_structure(&self, other: &Self) -> bool {
        self.fields == other.fields
            && self.values.len() == other.values.len()
            && self
                .field_parts()
                .iter()
                .zip(other.field_parts().iter())
                .all(|(a, b)| a.same_structure(b))
    }

    fn align(self, template: &Self, id: NodeId) -> Result<Self, EmitError> {
        if self.values.len() != template.values.len() {
            return Err(EmitError::at(id, "scan: step changes the state structure"));
        }
        if self.fields == template.fields {
            return Ok(self);
        }
        let (Some(fields), Some(expected)) = (&self.fields, &template.fields) else {
            return Err(EmitError::at(id, "scan: step changes the state structure"));
        };
        if fields.len() != expected.len() {
            return Err(EmitError::at(id, "scan: step changes the state fields"));
        }
        let parts = self.field_parts();
        let parts = expected
            .iter()
            .map(|name| {
                fields
                    .iter()
                    .position(|f| f == name)
                    .map(|i| parts[i].clone())
                    .ok_or_else(|| EmitError::at(id, "scan: step changes the state fields"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self::record(expected.clone(), parts))
    }

    fn with_values(&self, values: &[Value]) -> Self {
        Self {
            fields: self.fields.clone(),
            values: values.to_vec(),
            layout: self.layout.clone(),
        }
    }
}

impl Emitter<'_> {
    /// Dirac scores equality of one whole variate. Records retain their declared
    /// field order; tensor leaves contract cell axes, never callable batch axes.
    pub(crate) fn dirac_equal(&mut self, atom: NodeId, point: NodeId) -> Result<Value, EmitError> {
        let literal_fields = |id| {
            let id = self.resolve_tuple_projections(id);
            // A declared input replaces even a literal record's computation.
            // Its ABI field bindings take precedence over the source fields.
            if self.columns.keys().any(|(container, _)| *container == id) {
                return None;
            }
            match self.m.node(id) {
                Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if self.m.resolve(s) == "record") => {
                    Some(
                        c.named
                            .iter()
                            .map(|f| (f.name, f.value))
                            .collect::<Vec<_>>(),
                    )
                }
                _ => None,
            }
        };
        // Recurse before lowering fields: a nested literal record itself has no
        // tensor form. External flat records use the existing column bindings.
        if let (Some(atom), Some(point_fields)) = (literal_fields(atom), literal_fields(point)) {
            if !atom
                .iter()
                .map(|f| f.0)
                .eq(point_fields.iter().map(|f| f.0))
            {
                return Err(EmitError::at(
                    point,
                    "Dirac: record fields must match in order",
                ));
            }
            let mut equal = None;
            for ((_, a), (_, b)) in atom.into_iter().zip(point_fields) {
                let field = self.dirac_equal(a, b)?;
                equal = Some(match equal {
                    Some(previous) => self.and(&previous, &field),
                    None => field,
                });
            }
            return Ok(equal.unwrap_or_else(|| self.bool_value_const(true)));
        }
        let atom = self.lower_components(atom)?;
        let value = self.lower_components(point)?;
        if !atom.same_structure(&value) {
            return Err(EmitError::at(
                point,
                "Dirac: record fields must match in order",
            ));
        }
        let mut equal = None;
        for (a, b) in atom.values.iter().zip(&value.values) {
            let field = self.dirac_tensor_equal(point, a, b)?;
            equal = Some(match equal {
                Some(previous) => self.and(&previous, &field),
                None => field,
            });
        }
        Ok(equal.unwrap_or_else(|| self.bool_value_const(true)))
    }

    pub(crate) fn dirac_tensor_equal(
        &mut self,
        id: NodeId,
        atom: &Value,
        point: &Value,
    ) -> Result<Value, EmitError> {
        if self.cell_ty(atom) != self.cell_ty(point)
            || self.axes_of(atom).layers != self.axes_of(point).layers
        {
            return Err(EmitError::at(
                id,
                "Dirac: atom and point must have the same cell shape",
            ));
        }
        let (atom, point) = self.broadcast_batches(atom, point);
        let eq = self.compare("EQ", &point, &atom);
        Ok(if matches!(self.cell_ty(&eq), MlirTy::Scalar) {
            eq
        } else {
            self.reduce_boolean("stablehlo.and", "true", &eq)
        })
    }

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
            return Ok(Components::flat(None, vec![self.typed_axes(id, value)]));
        }
        let mut resolved = id;
        for _ in 0..64 {
            let next = self.resolve_ref_one(resolved);
            if next == resolved {
                break;
            }
            resolved = next;
        }
        if let Some(Type::Tuple(items)) = self.type_of(id) {
            let values = (0..items.len())
                .map(|i| match self.columns.get(&(resolved, i.to_string())) {
                    Some(Lowered::Tensor(value)) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            if let Some(values) = values {
                return Ok(Components::flat(None, values));
            }
            if let Node::Call(c) = self.m.node(resolved)
                && matches!(c.head, CallHead::Builtin(h) if self.m.resolve(h) == "tuple")
            {
                let values = c
                    .args
                    .iter()
                    .map(|&arg| {
                        self.lower_node(arg)
                            .map(|value| self.typed_axes(arg, value))
                    })
                    .collect::<Result<_, _>>()?;
                return Ok(Components::flat(None, values));
            }
        }
        if let Some(
            Type::Record(fields)
            | Type::Table {
                columns: fields, ..
            },
        ) = self.type_of(id)
        {
            let fields = fields.iter().map(|(name, _)| *name).collect::<Vec<_>>();
            if let Node::Call(c) = self.m.node(resolved)
                && let CallHead::Builtin(head) = c.head
                && matches!(self.m.resolve(head), "scan" | "reduce")
            {
                let reduce = self.m.resolve(head) == "reduce";
                return self.lower_sequence(resolved, &c.args, reduce);
            }
            let bound = fields.iter().all(|&name| {
                self.columns
                    .contains_key(&(resolved, self.m.resolve(name).to_owned()))
            });
            if !bound && let Some(entries) = self.literal_fields(resolved) {
                let entries = entries.to_vec();
                let mut parts = Vec::with_capacity(entries.len());
                for entry in &entries {
                    parts.push(match self.lower_value(entry.value)? {
                        Lowered::Tensor(value) => {
                            Lowered::Tensor(self.typed_axes(entry.value, value))
                        }
                        record => record,
                    });
                }
                let fields = entries.iter().map(|entry| entry.name).collect();
                return Ok(Components::record(fields, parts));
            }
            let Lowered::Record(lowered) = self.lower_value(id)? else {
                unreachable!("a record-typed value lowers to a record")
            };
            let parts = lowered.into_iter().map(|(_, part)| part).collect();
            return Ok(Components::record(fields, parts));
        }
        let value = self.lower_node(id)?;
        Ok(Components::flat(None, vec![self.typed_axes(id, value)]))
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
        // A body can reach a scan through a self-reference. Its call frame
        // must retain the mapped arguments while that binding is lowered.
        let call_frame = std::mem::replace(&mut self.call_frame, self.broadcast_frame.clone());
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
                if matches!(types[index], Type::Tuple(_)) {
                    for (i, value) in arg.values.iter().enumerate() {
                        self.bind_column(node, i.to_string(), value.clone());
                    }
                } else if let Some(fields) = &arg.fields {
                    for (&field, part) in fields.iter().zip(arg.field_parts()) {
                        self.bind_field(node, self.m.resolve(field).to_owned(), part);
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
        self.call_frame = call_frame;
        result
    }

    pub(super) fn sequence_field(
        &mut self,
        args: &[NodeId],
        base: i64,
    ) -> Option<Result<Value, EmitError>> {
        let [container, selector] = <[NodeId; 2]>::try_from(args).ok()?;
        if matches!(self.type_of(container), Some(Type::Tuple(_)))
            && self
                .columns
                .contains_key(&(self.resolve_ref_one(container), "0".to_owned()))
        {
            let Node::Lit(Scalar::Int(index)) = self.m.node(selector) else {
                return None;
            };
            let index = usize::try_from(index - base).ok()?;
            return Some(self.lower_components(container).and_then(|parts| {
                parts
                    .values
                    .get(index)
                    .cloned()
                    .ok_or_else(|| EmitError::at(container, "tuple index out of range"))
            }));
        }
        None
    }

    pub(super) fn lower_sequence(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        reduce: bool,
    ) -> Result<Components, EmitError> {
        let frame = self.broadcast_frame.clone();
        let previous = std::mem::replace(&mut self.call_frame, frame);
        let result = self.lower_sequence_in_frame(id, args, reduce);
        self.call_frame = previous;
        result
    }

    fn lower_sequence_in_frame(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        reduce: bool,
    ) -> Result<Components, EmitError> {
        if let Node::Call(call) = self.m.node(id)
            && !call.named.is_empty()
        {
            return Err(EmitError::at(
                id,
                "scan/reduce: keyword additional arguments are not lowered to StableHLO",
            ));
        }
        // `scan` traverses every sequence after `init` together (spec §04
        // "Additional arguments"); `reduce` holds its additional arguments constant.
        let (function, init, sequences, parameters) = match (reduce, args) {
            (false, [function, init, sequences @ ..]) if !sequences.is_empty() => {
                (*function, Some(*init), sequences, &[][..])
            }
            (true, [function, xs, parameters @ ..]) => {
                (*function, None, std::slice::from_ref(xs), parameters)
            }
            _ => {
                return Err(EmitError::at(
                    id,
                    "sequence callback: invalid argument count",
                ));
            }
        };
        let item_types = sequences
            .iter()
            .map(|&sequence| self.sequence_item_type(sequence))
            .collect::<Result<Vec<_>, _>>()?;
        let state_type = if reduce {
            item_types[0].clone()
        } else {
            self.sequence_item_type(id)?
        };
        let (body, entries) = self.callable_parts(function)?;
        if parameters.len() > 1 || entries.len() != 1 + sequences.len() + parameters.len() {
            return Err(EmitError::at(
                id,
                "scan: step must take state, one input per sequence and, for reduce, an optional parameter",
            ));
        }
        let parameter = parameters
            .first()
            .map(|&node| {
                let ty = self
                    .type_of(node)
                    .cloned()
                    .ok_or_else(|| EmitError::at(node, "scan: unresolved parameter type"))?;
                Ok((self.lower_components(node)?, ty))
            })
            .transpose()?;
        let seqs = sequences
            .iter()
            .map(|&sequence| self.lower_components(sequence))
            .collect::<Result<Vec<_>, _>>()?;
        // A one-dimensional array or table is traversed; any other input is a
        // constant held at every step (spec §04 "Additional arguments").
        let traversed: Vec<bool> = sequences
            .iter()
            .map(|&sequence| {
                matches!(
                    self.type_of(sequence),
                    Some(Type::Array { .. } | Type::TVector { .. } | Type::Table { .. })
                )
            })
            .collect();
        if !traversed.iter().any(|&t| t) {
            return Err(EmitError::at(
                id,
                "scan: needs at least one array or table input",
            ));
        }
        let mut lengths: Vec<Option<u64>> = Vec::with_capacity(seqs.len());
        for (xs, &is_sequence) in seqs.iter().zip(&traversed) {
            if !is_sequence {
                lengths.push(None);
                continue;
            }
            let Some(Some(len)) = xs
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
                shape(&v.ty).get(self.batch_rank(v)) != Some(&Some(len))
                    || shape(&v.ty).iter().any(Option::is_none)
                    || self.axes_of(v).layers.first() != Some(&1)
            }) {
                return Err(EmitError::at(
                    id,
                    "scan: expected a vector or equal-length table columns",
                ));
            }
            lengths.push(Some(len));
        }
        // Length-one sequences repeat; every other length must agree (§04).
        let mut others = lengths.iter().copied().flatten().filter(|&len| len != 1);
        let n = others.next().unwrap_or(1);
        if others.any(|len| len != n) {
            return Err(EmitError::at(id, "scan: sequences differ in length"));
        }
        let xs = &seqs[0];
        let mut state = if let Some(init) = init {
            self.lower_components(init)?
        } else {
            if n == 0 {
                return Err(EmitError::at(id, "reduce: input must be non-empty"));
            }
            let zero = self.int_value_const(0);
            let values = xs
                .values
                .iter()
                .map(|value| self.scan_slice(value, &zero))
                .collect::<Vec<_>>();
            xs.with_values(&values)
        };
        if let Some(fields) = &state.fields {
            let cached = fields
                .iter()
                .map(|&field| {
                    self.columns
                        .get(&(id, self.m.resolve(field).to_owned()))
                        .cloned()
                })
                .collect::<Option<Vec<_>>>();
            if let Some(parts) = cached {
                return Ok(Components::record(fields.clone(), parts));
            }
        } else if let Some(value) = self.memo.get(&id) {
            return Ok(state.with_values(std::slice::from_ref(value)));
        }

        let carry_types = match (self.type_of(id), &state.fields) {
            (Some(Type::Table { columns, .. } | Type::Record(columns)), Some(fields)) => {
                let mut leaves = Vec::with_capacity(state.values.len());
                for name in fields {
                    let ty = columns
                        .iter()
                        .find(|(field, _)| field == name)
                        .map(|(_, ty)| ty)
                        .ok_or_else(|| EmitError::at(id, "scan: unresolved state field"))?;
                    leaf_types(ty, &mut leaves);
                }
                if leaves.len() != state.values.len() {
                    return Err(EmitError::at(id, "scan: step changes the state structure"));
                }
                leaves
            }
            (Some(ty), None) if reduce => vec![ty.clone()],
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
        for value in state.values.iter().filter(|_| !reduce) {
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
        if n > u64::from(reduce) {
            let counter = self.int_value_const(i64::from(reduce));
            let mut inits = vec![counter];
            for (sequence, _) in seqs.iter().zip(&traversed).filter(|(_, t)| **t) {
                inits.extend(sequence.values.clone());
            }
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
                    let mut sequence_items = Vec::with_capacity(seqs.len());
                    let mut at = 1;
                    for (sequence, &len) in seqs.iter().zip(&lengths) {
                        let Some(len) = len else {
                            sequence_items.push(sequence.clone());
                            continue;
                        };
                        let index = if len == 1 {
                            e.int_value_const(0)
                        } else {
                            args[0].clone()
                        };
                        let mut items = Vec::with_capacity(sequence.values.len());
                        for (value, original) in args[at..].iter().zip(&sequence.values) {
                            let value = e.axes_view(value, e.axes_of(original));
                            items.push(e.scan_slice(&value, &index));
                        }
                        at += sequence.values.len();
                        sequence_items.push(sequence.with_values(&items));
                    }
                    let mut carries = Vec::with_capacity(state.values.len());
                    for (value, original) in args[start..out_start].iter().zip(&state.values) {
                        carries.push(e.axes_view(value, e.axes_of(original)));
                    }
                    let mut arguments = vec![state.with_values(&carries)];
                    arguments.extend(sequence_items);
                    let mut types = vec![state_type.clone()];
                    types.extend(item_types.iter().cloned());
                    if let Some((value, ty)) = &parameter {
                        arguments.push(value.clone());
                        types.push(ty.clone());
                    }
                    let output = e
                        .call_components(body, &entries, &arguments, &types)?
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
            buffers = if reduce {
                results[start..out_start].to_vec()
            } else {
                results[out_start..].to_vec()
            };
        } else if reduce {
            buffers = state.values.clone();
        }
        for (buffer, value) in buffers.iter_mut().zip(&state.values) {
            let mut axes = self.axes_of(value);
            if !reduce {
                axes.layers.insert(0, 1);
            }
            *buffer = self.axes_view(buffer, axes);
        }
        let result = state.with_values(&buffers);
        if let Some(fields) = &result.fields {
            for (&field, part) in fields.iter().zip(result.field_parts()) {
                self.bind_field(id, self.m.resolve(field).to_owned(), part);
            }
        } else {
            self.bind(id, buffers[0].clone());
        }
        Ok(result)
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

/// The tensor leaf types of `ty`, depth-first in declared field order.
fn leaf_types(ty: &Type, out: &mut Vec<Type>) {
    match ty {
        Type::Record(fields) => {
            for (_, field) in fields.iter() {
                leaf_types(field, out);
            }
        }
        ty => out.push(ty.clone()),
    }
}
