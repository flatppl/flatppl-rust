//! Callable state is flattened into tensor leaves. A scan emits one while,
//! with a fixed state type and an output buffer for each leaf.

use flatppl_core::Call;

use super::batching::tensor;
use super::*;

enum SequenceOutputs<'a> {
    Full,
    Terminal,
    Histories(&'a [Symbol]),
}

#[derive(Clone)]
pub(super) struct Components {
    pub(super) fields: Option<Vec<Symbol>>,
    /// The tensor leaves, depth-first in declared field order.
    pub(super) values: Vec<Value>,
    /// One structure per record field or tuple item when one is a record;
    /// `None` when every member is one leaf. Its tensors are placeholders: the
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
        Self::from_parts(Some(fields), parts)
    }

    fn from_parts(fields: Option<Vec<Symbol>>, parts: Vec<Lowered>) -> Self {
        // Tuple indices count whole members, including nested records, rather
        // than positions in the tensor leaves used to carry callback arguments.
        let mut values = Vec::new();
        for part in parts.iter().cloned() {
            part.leaves(&mut values);
        }
        let nested = parts.iter().any(|part| matches!(part, Lowered::Record(_)));
        Self {
            fields,
            values,
            layout: nested.then_some(parts),
        }
    }

    /// One value per record field or tuple item, rebuilt from the leaves.
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
        if self.fields.is_some() || self.layout.is_some() || self.values.len() != 1 {
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

    pub(super) fn align(self, template: &Self, id: NodeId) -> Result<Self, EmitError> {
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

    pub(super) fn with_values(&self, values: &[Value]) -> Self {
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
            let parts = (0..items.len())
                .map(|i| self.columns.get(&(resolved, i.to_string())).cloned())
                .collect::<Option<Vec<_>>>();
            if let Some(parts) = parts {
                return Ok(Components::from_parts(None, parts));
            }
            if let Node::Call(c) = self.m.node(resolved)
                && matches!(c.head, CallHead::Builtin(h) if self.m.resolve(h) == "tuple")
            {
                let parts = c
                    .args
                    .iter()
                    .map(|&arg| {
                        self.lower_value(arg).map(|part| match part {
                            Lowered::Tensor(value) => Lowered::Tensor(self.typed_axes(arg, value)),
                            record => record,
                        })
                    })
                    .collect::<Result<_, _>>()?;
                return Ok(Components::from_parts(None, parts));
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
                    for (i, part) in arg.field_parts().into_iter().enumerate() {
                        self.bind_field(node, i.to_string(), part);
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
    ) -> Option<Result<Lowered, EmitError>> {
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
                    .field_parts()
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
        self.lower_sequence_outputs(id, args, reduce, SequenceOutputs::Full)
    }

    fn in_scan_demand_scope(&self) -> bool {
        self.scan_demand.is_some()
            && self.region_depth == 0
            && self.scoped_inputs.is_empty()
            && self.broadcast_frame.is_empty()
            && self.call_frame.is_empty()
            && !self.host_batch
            && !self.fold_design_scans
    }

    fn has_scan_operand(&self, operand: NodeId) -> bool {
        matches!(
            self.m.node(self.resolve_ref_one(operand)),
            Node::Call(call) if matches!(call.head, CallHead::Builtin(head)
                if self.m.resolve(head) == "scan")
        )
    }

    fn scan_output_demand(&self) -> &flatppl_optimise::ScanOutputDemand {
        self.scan_demand
            .as_ref()
            .expect("checked scan demand scope")
            .get_or_init(|| flatppl_optimise::scan_output_demand_with_rules(self.m, &self.rules))
    }

    pub(super) fn lower_scan_terminal(
        &mut self,
        reader: NodeId,
        args: &[NodeId],
    ) -> Option<Result<Value, EmitError>> {
        if !self.in_scan_demand_scope()
            || !matches!(args, [operand, _] if self.has_scan_operand(*operand))
        {
            return None;
        }
        let scan = self.scan_output_demand().terminal_scan(reader)?;
        if let Some(value) = self.scan_terminals.get(&scan) {
            return Some(Ok(value.clone()));
        }
        // A full buffer produced by another lowering path still uses the
        // ordinary indexing path; it is never a terminal cache entry.
        if self.memo.contains_key(&scan) {
            return None;
        }
        let Node::Call(call) = self.m.node(scan) else {
            unreachable!("terminal scan proof");
        };
        Some(
            self.lower_sequence_outputs(scan, &call.args, false, SequenceOutputs::Terminal)
                .and_then(|output| output.tensor(scan))
                .inspect(|value| {
                    self.scan_terminals.insert(scan, value.clone());
                }),
        )
    }

    pub(super) fn lower_scan_history_map(
        &mut self,
        reader: NodeId,
        args: &[NodeId],
    ) -> Option<Result<Value, EmitError>> {
        if !self.in_scan_demand_scope()
            || !matches!(args, [_, operand] if self.has_scan_operand(*operand))
        {
            return None;
        }
        let (scan, field) = self.scan_output_demand().history_map(reader)?;
        let field = self.m.resolve(field).to_owned();
        Some(
            self.lower_field(scan, &field)
                .and_then(|value| match value {
                    Lowered::Tensor(value) => Ok(value),
                    Lowered::Record(_) => Err(records::record_in_tensor_position(reader, &field)),
                }),
        )
    }

    pub(super) fn lower_scan_histories(
        &mut self,
        scan: NodeId,
        args: &[NodeId],
    ) -> Result<Components, EmitError> {
        if self.in_scan_demand_scope()
            && let Some(fields) = self
                .scan_output_demand()
                .histories(scan)
                .map(<[Symbol]>::to_vec)
        {
            self.lower_sequence_outputs(scan, args, false, SequenceOutputs::Histories(&fields))
        } else {
            self.lower_sequence(scan, args, false)
        }
    }

    fn lower_sequence_outputs(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        reduce: bool,
        outputs: SequenceOutputs<'_>,
    ) -> Result<Components, EmitError> {
        let frame = self.broadcast_frame.clone();
        let previous = std::mem::replace(&mut self.call_frame, frame);
        let result = self.lower_sequence_in_frame(id, args, reduce, outputs);
        self.call_frame = previous;
        result
    }

    fn lower_sequence_in_frame(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        reduce: bool,
        outputs: SequenceOutputs<'_>,
    ) -> Result<Components, EmitError> {
        if let Node::Call(call) = self.m.node(id)
            && !call.named.is_empty()
        {
            return Err(EmitError::at(
                id,
                "scan/reduce: keyword additional arguments are not lowered to StableHLO",
            ));
        }
        // `scan(f, init, xs, p)` traverses `xs` and passes `p` whole, held at every
        // step as a constant input; `reduce(f, xs, p)` passes `p` as its parameter
        // (spec §04 "Explicit callback parameters").
        let (function, init, sequences, parameters) = match (reduce, args) {
            (false, [function, init, sequences @ ..]) if (1..=2).contains(&sequences.len()) => {
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
            .enumerate()
            .map(|(i, &sequence)| {
                if i == 0 {
                    self.sequence_item_type(sequence)
                } else {
                    self.type_of(sequence)
                        .cloned()
                        .ok_or_else(|| EmitError::at(sequence, "scan: unresolved parameter type"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        fn real_state(ty: &Type) -> bool {
            match ty {
                Type::Scalar(flatppl_core::ScalarType::Real) => true,
                Type::Array { elem, .. } => real_state(elem),
                Type::Record(fields) => fields.iter().all(|(_, ty)| real_state(ty)),
                _ => false,
            }
        }
        let state_type = if reduce {
            item_types[0].clone()
        } else {
            let inferred = self.sequence_item_type(id)?;
            match (&inferred, init.and_then(|node| self.type_of(node))) {
                // A nested callback can remain deferred even though its
                // initial real-valued state is fully typed. The loop checks
                // the returned structure and tensor shape below.
                (Type::Deferred, Some(ty)) if real_state(ty) => ty.clone(),
                _ => inferred,
            }
        };
        let (body, entries) = self.callable_parts(function)?;
        if parameters.len() > 1 || entries.len() != 1 + sequences.len() + parameters.len() {
            return Err(EmitError::at(
                id,
                "scan: step must take the state, one element of `xs` and an optional parameter",
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
        // Only `xs` is traversed; a `p` is a constant held at every step, whatever
        // its shape (spec §04 "Explicit callback parameters").
        let traversed: Vec<bool> = (0..sequences.len()).map(|i| i == 0).collect();
        if !matches!(
            self.type_of(sequences[0]),
            Some(Type::Array { .. } | Type::TVector { .. } | Type::Table { .. })
        ) {
            return Err(EmitError::at(
                sequences[0],
                "scan: `xs` must be an array or table",
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
        let n = lengths[0].expect("`xs` is traversed");
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

        let output_type = match self.type_of(id) {
            Some(Type::Deferred) if real_state(&state_type) => Some(&state_type),
            ty => ty,
        };
        let carry_types = match (output_type, &state.fields) {
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
            (Some(ty), None)
                if matches!(self.type_of(id), Some(Type::Deferred)) && real_state(ty) =>
            {
                vec![ty.clone()]
            }
            (Some(Type::Array { elem, .. }), None) => vec![(**elem).clone()],
            _ => return Err(EmitError::at(id, "scan: unresolved state type")),
        };
        // An rng state rides in the carry with no output buffer: only the state
        // after the last step is kept (spec §07 "Random number generation").
        let keys: Vec<bool> = state.values.iter().map(|v| v.ty == MlirTy::Key).collect();
        let terminal = matches!(outputs, SequenceOutputs::Terminal);
        let selected = match outputs {
            SequenceOutputs::Histories(fields)
                if state.layout.is_none() && !keys.iter().any(|&key| key) =>
            {
                state
                    .fields
                    .as_ref()
                    .filter(|names| names.len() == state.values.len())
                    .map(|names| (names, fields))
            }
            _ => None,
        };
        let histories: Vec<_> = keys
            .iter()
            .enumerate()
            .filter_map(|(i, &key)| {
                (!reduce
                    && !key
                    && selected.is_none_or(|(names, fields)| fields.contains(&names[i])))
                .then_some(i)
            })
            .collect();
        if keys.iter().any(|&k| k) && (reduce || n == 0 || !self.call_frame.is_empty()) {
            return Err(EmitError::at(
                id,
                "scan: an rng state in the carry is lowered only for an unmapped, non-empty scan",
            ));
        }
        for (value, ty) in state.values.iter_mut().zip(carry_types) {
            if value.ty == MlirTy::Key {
                if ty != Type::RngState {
                    return Err(EmitError::at(
                        id,
                        "scan: initial value does not fit the state type",
                    ));
                }
                continue;
            }
            let (shape, kind) = crate::types::mlir_type_of_ty(id, &ty, self.dtype)?;
            // A fixed selection can resolve a dynamic inferred extent at emission.
            let actual = self.cell_ty(value);
            let fits = match (&actual, &shape) {
                (MlirTy::Ranked(actual), MlirTy::Ranked(expected)) => {
                    actual.len() == expected.len()
                        && actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| b.is_none() || a == b)
                }
                _ => actual == shape,
            };
            if !fits || elem_rank(value.elem) > elem_rank(kind) {
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
        let fixed = if self.fold_design_scans
            && !reduce
            && let [seq] = seqs.as_slice()
            && let [xs] = seq.values.as_slice()
            && let [init] = state.values.as_slice()
            && state.fields.is_none()
            && seq.fields.is_none()
        {
            self.fixed_add_scan(body, &entries, init, xs)
                .map(|out| vec![out])
        } else {
            None
        };
        // A profile without serial scans replaces the loop when the step is
        // affine in the carry.
        let computed = if fixed.is_some() {
            fixed
        } else if reduce || self.target.serial_scans() || keys.iter().any(|&k| k) {
            None
        } else {
            let mut types = vec![state_type.clone()];
            types.extend(item_types.iter().cloned());
            self.try_affine_scan(&super::affine_scan::AffineScan {
                id,
                body,
                entries: &entries,
                state: &state,
                types: &types,
                seqs: &seqs,
                lengths: &lengths,
                n,
            })
        };
        let mut buffers = Vec::with_capacity(state.values.len());
        for &i in histories.iter().filter(|_| computed.is_none()) {
            let value = &state.values[i];
            let mut dims = shape(&value.ty).to_vec();
            dims.insert(self.batch_rank(value), Some(if terminal { 1 } else { n }));
            let zero = match value.elem {
                ElemKind::Real => self.scalar(0.0),
                ElemKind::Complex => {
                    let zero = self.scalar(0.0);
                    self.convert(&zero, ElemKind::Complex)
                }
                ElemKind::Int => self.int_value_const(0),
                ElemKind::Bool => self.bool_value_const(false),
            };
            let buffer = self.broadcast_in_dim(&zero, &[], tensor(dims));
            let mut axes = self.axes_of(value);
            axes.layers.insert(0, 1);
            buffers.push(self.axes_view(&buffer, axes));
        }
        // Enzyme indexes its reverse-pass tapes by a counter that starts at
        // zero. `reduce` begins at the second item, so it runs `n - 1` steps
        // and reads the item after the counter.
        let first = u64::from(reduce);
        if let Some(values) = computed {
            debug_assert!(!terminal && selected.is_none());
            buffers = values;
        } else if n > first {
            let width = if reduce {
                1
            } else {
                self.scan_block_width(body, &state, xs, n)
            };
            debug_assert!(!terminal || width == 1);
            let index_range = if !reduce
                && !self.target.serial_scans()
                && !self.host_batch
                && self.region_depth == 0
                && self.call_frame.is_empty()
                && self.runtime_batch_rank == 0
                && self.rules.enables_ruleset("ordered-scan-index")
                && xs.fields.is_none()
                && xs.layout.is_none()
                && let [value] = xs.values.as_slice()
            {
                self.consecutive_scan_index(value, n)
                    .and_then(|(first, last)| {
                        flatppl_optimise::plan_ordered_scan_index_with_rules(
                            first,
                            last,
                            n,
                            width,
                            &self.rules,
                        )
                    })
            } else {
                None
            };
            let counter = self.int_value_const(0);
            let mut inits = vec![counter];
            if index_range.is_none() {
                inits.extend(xs.values.iter().cloned());
            }
            let start = inits.len();
            inits.extend(state.values.clone());
            let out_start = inits.len();
            inits.extend(buffers);
            let types = inits
                .iter()
                .map(|v| v.ty.render(self.dtype, v.elem))
                .collect::<Vec<_>>();
            let advance = |e: &mut Self,
                           args: &[Value],
                           index: &Value|
             -> Result<Vec<Value>, EmitError> {
                let mut sequence_items = Vec::with_capacity(seqs.len());
                let mut scan_index = None;
                let item = if reduce {
                    let one = e.int_value_const(1);
                    e.add(index, &one)
                } else {
                    index.clone()
                };
                for (sequence, &len) in seqs.iter().zip(&lengths) {
                    let Some(len) = len else {
                        sequence_items.push(sequence.clone());
                        continue;
                    };
                    let index = if len == 1 {
                        e.int_value_const(0)
                    } else {
                        item.clone()
                    };
                    if let Some((first, last)) = index_range {
                        let item = e.scan_index_value(&index, first);
                        scan_index = Some((item.ssa.clone(), first, last));
                        sequence_items.push(sequence.with_values(&[item]));
                        continue;
                    }
                    let items = args[1..start]
                        .iter()
                        .zip(&sequence.values)
                        .map(|(value, original)| {
                            let value = e.axes_view(value, e.axes_of(original));
                            e.scan_slice(&value, &index)
                        })
                        .collect::<Vec<_>>();
                    sequence_items.push(sequence.with_values(&items));
                }
                let mut carries = Vec::with_capacity(state.values.len());
                for (value, original) in args[start..out_start].iter().zip(&state.values) {
                    carries.push(if original.ty == MlirTy::Key {
                        value.clone()
                    } else {
                        e.axes_view(value, e.axes_of(original))
                    });
                }
                let mut arguments = vec![state.with_values(&carries)];
                arguments.extend(sequence_items);
                let mut types = vec![state_type.clone()];
                types.extend(item_types.iter().cloned());
                if let Some((value, ty)) = &parameter {
                    arguments.push(value.clone());
                    types.push(ty.clone());
                }
                let previous_index = std::mem::replace(&mut e.scan_index, scan_index);
                let output = e.call_components(body, &entries, &arguments, &types);
                e.scan_index = previous_index;
                let output = output?.align(&state, id)?;
                let mut next = Vec::with_capacity(args.len());
                let one = e.int_value_const(1);
                next.push(e.add(&args[0], &one));
                next.extend_from_slice(&args[1..start]);
                for (value, expected) in output.values.iter().zip(&state.values) {
                    if expected.ty == MlirTy::Key || value.ty == MlirTy::Key {
                        if value.ty != expected.ty {
                            return Err(EmitError::at(id, "scan: step changes the state type"));
                        }
                        next.push(value.clone());
                        continue;
                    }
                    let frame = e.call_frame.clone();
                    let value = e.finish_broadcast(value, &frame, frame.len());
                    if value.ty != expected.ty || elem_rank(value.elem) > elem_rank(expected.elem) {
                        return Err(EmitError::at(id, "scan: step changes the state type"));
                    }
                    next.push(e.convert(&value, expected.elem));
                }
                Ok(next)
            };
            let block = |e: &mut Self,
                         args: &[Value],
                         index: &Value,
                         count: u64|
             -> Result<Vec<Value>, EmitError> {
                let mut next = args[..out_start].to_vec();
                let mut rows = vec![Vec::new(); args.len() - out_start];
                for offset in 0..count {
                    let item = if offset == 0 {
                        index.clone()
                    } else {
                        let offset = e.int_value_const(offset as i64);
                        e.add(index, &offset)
                    };
                    next = advance(e, &next, &item)?;
                    for (rows, &leaf) in rows.iter_mut().zip(&histories) {
                        rows.push(next[start + leaf].clone());
                    }
                }
                for (buffer, rows) in args[out_start..].iter().zip(rows) {
                    let update = if let [row] = rows.as_slice() {
                        // A one-row history write retains Enzyme's per-step cotangent paths.
                        let slot = terminal.then(|| e.int_value_const(0));
                        e.scan_update(buffer, row, slot.as_ref().unwrap_or(index))
                    } else {
                        let block = e.vector(&rows);
                        e.update_scan_rows(buffer, &block, index, e.batch_rank(&block))
                    };
                    next.push(update);
                }
                Ok(next)
            };
            // The loop counts blocks from zero for Enzyme's reverse-pass tapes.
            // A short tail executes only the remaining callbacks, in order.
            let steps = n - first;
            let mut results = self.try_while_loop(
                &inits,
                &types,
                |e, args| {
                    let limit = e.int_value_const((steps / width) as i64);
                    e.compare("LT", &args[0], &limit)
                },
                |e, args| {
                    let index = if width == 1 {
                        args[0].clone()
                    } else {
                        let width = e.int_value_const(width as i64);
                        e.mul(&args[0], &width)
                    };
                    let mut result = block(e, args, &index, width)?;
                    let one = e.int_value_const(1);
                    result[0] = e.add(&args[0], &one);
                    Ok(result)
                },
            )?;
            if !steps.is_multiple_of(width) {
                let tail = self.int_value_const((steps - steps % width) as i64);
                results = block(self, &results, &tail, steps % width)?;
            }
            if terminal {
                let buffer = &results[out_start];
                let limits: Vec<_> = shape(&buffer.ty).iter().map(|dim| dim.unwrap()).collect();
                let row = self.slice(
                    buffer,
                    &vec![0; limits.len()],
                    &limits,
                    &vec![1; limits.len()],
                );
                let carry = &state.values[0];
                let value = self.reshape_axes(&row, carry.ty.clone(), self.axes_of(carry));
                return Ok(Components::flat(None, vec![value]));
            }
            buffers = if reduce {
                results[start..out_start].to_vec()
            } else if selected.is_some() {
                results[out_start..].to_vec()
            } else {
                let mut columns = results[out_start..].iter().cloned();
                results[start..out_start]
                    .iter()
                    .zip(&keys)
                    .map(|(last, &key)| {
                        if key {
                            last.clone()
                        } else {
                            columns.next().expect("one buffer per tensor leaf")
                        }
                    })
                    .collect()
            };
            for (last, &key) in buffers.iter().zip(&keys).filter(|_| selected.is_none()) {
                if key {
                    // The next draw after the scan reads its final state.
                    self.set_cur_key(last.clone());
                    self.final_scan_keys.insert(last.ssa.clone(), n);
                }
            }
        } else if reduce {
            buffers = state.values.clone();
        }
        if let Some((names, _)) = selected {
            state = Components::flat(
                Some(histories.iter().map(|&index| names[index]).collect()),
                histories
                    .iter()
                    .map(|&index| state.values[index].clone())
                    .collect(),
            );
        }
        for (buffer, value) in buffers.iter_mut().zip(&state.values) {
            if value.ty == MlirTy::Key {
                continue;
            }
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

    /// Small real carries have little device parallelism. Group ordered steps
    /// per GPU loop to amortize launch overhead without reassociating them.
    fn scan_block_width(
        &self,
        body: NodeId,
        state: &Components,
        xs: &Components,
        length: u64,
    ) -> u64 {
        let compact_state = |v: &Value| {
            v.elem == ElemKind::Real
                && (v.ty == MlirTy::Scalar
                    || matches!(&v.ty, MlirTy::Ranked(dims) if matches!(dims.as_slice(), [Some(1..=8)])))
        };
        let compact_items = |v: &Value| matches!(&v.ty, MlirTy::Ranked(dims) if matches!(dims.as_slice(), [_] | [_, Some(1..=8)]));
        if self.target.serial_scans()
            || length < 32
            || !self.call_frame.is_empty()
            || self.region_depth != 0
            || state.values.len() > 4
            || xs.values.len() > 4
            || !state.values.iter().all(compact_state)
            || !xs.values.iter().all(compact_items)
        {
            return 1;
        }
        let mut allow_wide = length >= 128
            && state.fields.is_none()
            && xs.fields.is_none()
            && matches!(state.values.as_slice(), [value] if value.ty == MlirTy::Scalar)
            && matches!(xs.values.as_slice(), [value]
                if value.elem == ElemKind::Real
                    && matches!(&value.ty, MlirTy::Ranked(dims) if dims.len() == 1));
        // Bound repeated code and skip nested loops. Already-bound operands
        // are captured once in the enclosing function.
        const MAX_NODES: usize = 48;
        let mut pending = vec![body];
        let mut seen = HashSet::new();
        while let Some(id) = pending.pop() {
            if self.memo.contains_key(&id) || !seen.insert(id) {
                continue;
            }
            if seen.len() > MAX_NODES {
                return 1;
            }
            let node = self.m.node(id);
            if let Node::Call(call) = node {
                if call.args.len() + call.named.len() > MAX_NODES {
                    return 1;
                }
                if let CallHead::Builtin(head) = call.head
                    && matches!(
                        self.m.resolve(head),
                        "scan" | "reduce" | "builtin_sample" | "integral"
                    )
                {
                    return 1;
                }
                allow_wide = allow_wide && self.simple_scan_call(call);
            }
            node.for_each_child(|child| pending.push(child));
            let target = self.resolve_ref_one(id);
            if target != id {
                pending.push(target);
            }
        }
        if allow_wide {
            flatppl_optimise::ordered_scan_width_with_rules(length, seen.len(), &self.rules)
        } else {
            4
        }
    }

    /// Widen only elementary arithmetic and statically named field reads. In
    /// particular, indexing, checked operations and opaque callbacks keep the
    /// existing four-step plan.
    fn simple_scan_call(&self, call: &Call) -> bool {
        if !call.named.is_empty() || call.inputs.is_some() {
            return false;
        }
        let CallHead::Builtin(head) = call.head else {
            return false;
        };
        match (self.m.resolve(head), call.args.as_ref()) {
            ("add" | "sub" | "mul", [_, _]) | ("neg" | "sqrt", [_]) => true,
            ("pow", [_, exponent]) => matches!(self.m.node(*exponent), Node::Lit(Scalar::Int(2))),
            ("get" | "get_field", [_, key]) => {
                matches!(self.m.node(*key), Node::Lit(Scalar::Str(_)))
            }
            _ => false,
        }
    }

    /// A left scan of scalar integers with the literal callback `(a, x) -> a + x`.
    /// Every addition must fit the target's integer width. The final axis is
    /// traversed independently within each mapped lane.
    fn fixed_add_scan(
        &mut self,
        body: NodeId,
        entries: &[(Symbol, Ref)],
        init: &Value,
        xs: &Value,
    ) -> Option<Value> {
        let [(_, carry), (_, item)] = entries else {
            return None;
        };
        let Node::Call(call) = self.m.node(body) else {
            return None;
        };
        let [a, b] = call.args.as_ref() else {
            return None;
        };
        let (Node::Ref(a), Node::Ref(b)) = (self.m.node(*a), self.m.node(*b)) else {
            return None;
        };
        if !matches!(call.head, CallHead::Builtin(s) if self.resolve(s) == "add")
            || !call.named.is_empty()
            || !((a == carry && b == item) || (a == item && b == carry))
            || init.elem != ElemKind::Int
            || xs.elem != ElemKind::Int
        {
            return None;
        }
        let (last, prefix) = shape(&xs.ty).split_last()?;
        if prefix != shape(&init.ty)
            || self.batch_rank(xs) != prefix.len()
            || self.batch_rank(init) != prefix.len()
        {
            return None;
        }
        let width = usize::try_from((*last)?).ok()?;
        let cells = super::constants::element_count(&xs.ty)?;
        if width == 0 || cells == 0 || cells > super::constants::MAX_FOLDED_ELEMENTS {
            return None;
        }
        let mut values = Vec::with_capacity(cells);
        for lane in 0..cells / width {
            let Scalar::Int(mut carry) = self.fixed_scalar_at(init, lane)? else {
                return None;
            };
            for index in lane * width..(lane + 1) * width {
                let Scalar::Int(next) = self.fixed_scalar_at(xs, index)? else {
                    return None;
                };
                carry = carry.checked_add(next)?;
                if !self.int_fits(carry) {
                    return None;
                }
                values.push(Scalar::Int(carry));
            }
        }
        self.folded_constant(values, xs.ty.clone(), self.axes_of(xs))
    }

    fn consecutive_scan_index(&self, value: &Value, len: u64) -> Option<(i64, i64)> {
        if len < 2 || value.elem != ElemKind::Int || shape(&value.ty) != [Some(len)] {
            return None;
        }
        let values = self.constants.get(&value.ssa)?;
        let len = usize::try_from(len).ok()?;
        if values.len() != len || len > super::constants::MAX_FOLDED_ELEMENTS {
            return None;
        }
        let Scalar::Int(first) = values[0] else {
            return None;
        };
        let last = first.checked_add(i64::try_from(len - 1).ok()?)?;
        (self.int_fits(first)
            && self.int_fits(last)
            && values
                .iter()
                .zip(first..=last)
                .all(|(value, expect)| matches!(value, Scalar::Int(i) if *i == expect)))
        .then_some((first, last))
    }

    fn scan_index_value(&mut self, position: &Value, first: i64) -> Value {
        let first = self.int_value_const(first);
        let ty = position.ty.render(self.dtype, ElemKind::Int);
        // Even a constant tail position must retain the scan's runtime
        // out-of-bounds behavior when the full index range cannot be proved.
        let ssa = self.pure(format!(
            "stablehlo.add {}, {} : {ty}",
            position.ssa, first.ssa
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Int,
        }
    }

    pub(crate) fn scan_index_in_bounds(&self, index: &Value, base: i64, upper: u64) -> bool {
        self.scan_index.as_ref().is_some_and(|(ssa, first, last)| {
            ssa == &index.ssa && *first >= base && u64::try_from(*last).is_ok_and(|i| i <= upper)
        })
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
        self.update_scan_rows(buffer, &update, index, axis)
    }

    fn update_scan_rows(
        &mut self,
        buffer: &Value,
        update: &Value,
        index: &Value,
        axis: usize,
    ) -> Value {
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
pub(super) fn leaf_types(ty: &Type, out: &mut Vec<Type>) {
    match ty {
        Type::Record(fields) => {
            for (_, field) in fields.iter() {
                leaf_types(field, out);
            }
        }
        ty => out.push(ty.clone()),
    }
}
