//! Record values (spec §03 "Records" and "Tables"). A record lowers to a
//! [`Lowered::Record`] of its fields in declared order. A table is the record of
//! its columns, each column carrying the row axis first. A field access lowers
//! only the projected field when the producer allows it.

use super::*;

impl Emitter<'_> {
    /// The declared field names of a record- or table-typed node.
    pub(crate) fn record_fields(&self, id: NodeId) -> Option<Vec<String>> {
        match self.type_of(id) {
            Some(
                Type::Record(fields)
                | Type::Table {
                    columns: fields, ..
                },
            ) => Some(
                fields
                    .iter()
                    .map(|(name, _)| self.m.resolve(*name).to_owned())
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Lower `id` to a tensor, or to a record of its fields when its inferred
    /// type is a record or a table.
    pub(crate) fn lower_value(&mut self, id: NodeId) -> Result<Lowered, EmitError> {
        let Some(fields) = self.record_fields(id) else {
            return Ok(Lowered::Tensor(self.lower_node(id)?));
        };
        let mut lowered = Vec::with_capacity(fields.len());
        for name in fields {
            let value = self.lower_field(id, &name)?;
            lowered.push((name, value));
        }
        Ok(Lowered::Record(lowered))
    }

    /// `get(container, "field")` as `(container, field)` when `container` is a
    /// record or table: by inferred type, by a bound field, or by a literal
    /// producer.
    pub(super) fn record_projection(&self, args: &[NodeId]) -> Option<(NodeId, String)> {
        let [container, selector] = <[NodeId; 2]>::try_from(args).ok()?;
        let field = match self.m.node(selector) {
            Node::Lit(Scalar::Str(s)) => s.to_string(),
            Node::Const(sym) => self.m.resolve(*sym).to_owned(),
            _ => return None,
        };
        let record = self.record_fields(container).is_some()
            || self
                .columns
                .contains_key(&(self.column_container(container), field.clone()))
            || self
                .literal_fields(self.resolve_tuple_projections(container))
                .is_some();
        record.then_some((container, field))
    }

    /// The named entries of a `record(...)` or `table(...)` call.
    pub(super) fn literal_fields(&self, id: NodeId) -> Option<&[flatppl_core::NamedArg]> {
        match self.m.node(id) {
            Node::Call(c)
                if matches!(c.head, CallHead::Builtin(sym) if matches!(self.m.resolve(sym), "record" | "table"))
                    && !c.named.is_empty() =>
            {
                Some(&c.named)
            }
            _ => None,
        }
    }

    /// Lower field `name` of the record or table `container`. A bound field
    /// reads its binding. Otherwise only the producer's own field is lowered,
    /// so a projection of a literal record emits no sibling field.
    pub(crate) fn lower_field(
        &mut self,
        container: NodeId,
        name: &str,
    ) -> Result<Lowered, EmitError> {
        if let Some(value) = self.bound_field(container, name) {
            return Ok(value);
        }
        let resolved = self.resolve_tuple_projections(container);
        if let Some(value) = self.bound_field(resolved, name) {
            return Ok(value);
        }
        if let Some(entries) = self.literal_fields(resolved) {
            let entry = entries
                .iter()
                .find(|entry| self.m.resolve(entry.name) == name)
                .map(|entry| entry.value)
                .ok_or_else(|| unknown_field(container, name))?;
            return self.lower_value(entry);
        }
        let Node::Call(call) = self.m.node(resolved) else {
            return Err(no_field_lowering(container, name, "this value"));
        };
        let CallHead::Builtin(head) = call.head else {
            return Err(no_field_lowering(container, name, "a user call"));
        };
        let head = self.m.resolve(head).to_owned();
        let args = call.args.clone();
        match (head.as_str(), &args[..]) {
            // §03 "Tables": `table(r)` and `record(t)` convert between a table
            // and the record of its columns, which is one representation here.
            ("record" | "table", [inner]) => self.lower_field(*inner, name),
            ("get" | "get0", [inner, selector]) => {
                let base = i64::from(head == "get");
                self.lower_projected_field(resolved, *inner, *selector, base, name)
            }
            ("ifelse", [cond, a, b]) => {
                let cond = crate::ops::lower_predicate(self, *cond, "ifelse condition")?;
                let a = self.lower_field(*a, name)?;
                let b = self.lower_field(*b, name)?;
                self.select_lowered(resolved, &cond, a, b)
            }
            (head, [table])
                if crate::norms::TABLE_REDUCTIONS.contains(&head)
                    && matches!(self.type_of(*table), Some(Type::Table { .. })) =>
            {
                let column = self.lower_field(*table, name)?;
                self.reduce_columns(resolved, head, column)
            }
            ("broadcast", [callable, rest @ ..])
                if let Some(function) = self.resolve_functionof(*callable) =>
            {
                let parts = self.lower_broadcast_components(resolved, function, rest)?;
                let fields = parts
                    .fields
                    .clone()
                    .ok_or_else(|| no_field_lowering(container, name, "a broadcast"))?;
                let mut found = None;
                for (field, part) in fields.into_iter().zip(parts.field_parts()) {
                    let field = self.m.resolve(field).to_owned();
                    if field == name {
                        found = Some(part.clone());
                    }
                    self.bind_field(resolved, field, part);
                }
                found.ok_or_else(|| unknown_field(container, name))
            }
            ("scan" | "reduce", _) => {
                let parts = self.lower_sequence(resolved, &args, head == "reduce")?;
                let index = parts
                    .fields
                    .as_ref()
                    .and_then(|fields| fields.iter().position(|&f| self.m.resolve(f) == name))
                    .ok_or_else(|| unknown_field(container, name))?;
                Ok(parts.field_parts().swap_remove(index))
            }
            (head, _) => Err(no_field_lowering(
                container,
                name,
                &format!("a record-valued `{head}`"),
            )),
        }
    }

    /// §07 "Table reductions" over one column. §03 "Tables" lets a column be a
    /// table, and that column reduces column-wise in turn.
    fn reduce_columns(
        &mut self,
        id: NodeId,
        head: &str,
        column: Lowered,
    ) -> Result<Lowered, EmitError> {
        match column {
            Lowered::Tensor(column) => {
                crate::norms::reduce_column(self, id, head, column).map(Lowered::Tensor)
            }
            Lowered::Record(columns) => columns
                .into_iter()
                .map(|(name, column)| Ok((name, self.reduce_columns(id, head, column)?)))
                .collect::<Result<_, _>>()
                .map(Lowered::Record),
        }
    }

    fn bound_field(&self, container: NodeId, name: &str) -> Option<Lowered> {
        self.columns
            .get(&(self.column_container(container), name.to_owned()))
            .cloned()
    }

    /// Field `name` of `get(inner, selector)`, whose result is a record.
    fn lower_projected_field(
        &mut self,
        id: NodeId,
        inner: NodeId,
        selector: NodeId,
        base: i64,
        name: &str,
    ) -> Result<Lowered, EmitError> {
        match self.m.node(selector) {
            // `r.a.b`: the field of a record-valued field.
            Node::Lit(Scalar::Str(field)) => {
                let field = field.to_string();
                if let Some(entry) = self.literal_entry(inner, &field) {
                    return self.lower_field(entry, name);
                }
                match self.lower_field(inner, &field)? {
                    Lowered::Record(fields) => fields
                        .into_iter()
                        .find(|(f, _)| f == name)
                        .map(|(_, value)| value)
                        .ok_or_else(|| unknown_field(id, name)),
                    Lowered::Tensor(_) => Err(unknown_field(id, name)),
                }
            }
            // §03 "Tables": `t[i]` is row `i` as a record.
            _ if matches!(self.type_of(inner), Some(Type::Table { .. })) => {
                let column = self.lower_field(inner, name)?;
                self.index_rows(id, column, selector, base)
            }
            _ => Err(no_field_lowering(id, name, "this projection")),
        }
    }

    /// The literal producer entry for field `field` of `container`, if any.
    fn literal_entry(&self, container: NodeId, field: &str) -> Option<NodeId> {
        if self.bound_field(container, field).is_some() {
            return None;
        }
        let resolved = self.resolve_tuple_projections(container);
        self.literal_fields(resolved)?
            .iter()
            .find(|entry| self.m.resolve(entry.name) == field)
            .map(|entry| entry.value)
    }

    /// Select row `index` of every column of a (nested) table column.
    fn index_rows(
        &mut self,
        id: NodeId,
        column: Lowered,
        index: NodeId,
        base: i64,
    ) -> Result<Lowered, EmitError> {
        match column {
            Lowered::Tensor(value) => {
                let value = self.row_axes(value);
                crate::ops::get_outer(self, id, value, index, base).map(Lowered::Tensor)
            }
            Lowered::Record(fields) => fields
                .into_iter()
                .map(|(field, column)| Ok((field, self.index_rows(id, column, index, base)?)))
                .collect::<Result<_, _>>()
                .map(Lowered::Record),
        }
    }

    /// A column's row axis is a vector axis, whatever its cells hold.
    fn row_axes(&mut self, value: Value) -> Value {
        let mut axes = self.axes_of(&value);
        let rank = shape(&value.ty).len() - axes.batch;
        if axes.layers.first() == Some(&1) || rank == 0 {
            return value;
        }
        axes.layers = vec![1, rank - 1];
        self.axes_view(&value, axes)
    }

    fn select_lowered(
        &mut self,
        id: NodeId,
        cond: &Value,
        a: Lowered,
        b: Lowered,
    ) -> Result<Lowered, EmitError> {
        match (a, b) {
            (Lowered::Tensor(a), Lowered::Tensor(b)) => {
                Ok(Lowered::Tensor(self.select(cond, &a, &b)))
            }
            (Lowered::Record(a), Lowered::Record(b))
                if a.iter().map(|(f, _)| f).eq(b.iter().map(|(f, _)| f)) =>
            {
                a.into_iter()
                    .zip(b)
                    .map(|((field, a), (_, b))| Ok((field, self.select_lowered(id, cond, a, b)?)))
                    .collect::<Result<_, _>>()
                    .map(Lowered::Record)
            }
            _ => Err(EmitError::at(
                id,
                "ifelse: the two branches are records with different fields",
            )),
        }
    }
}

fn unknown_field(id: NodeId, name: &str) -> EmitError {
    EmitError::at(id, format!("the record has no field `{name}`"))
}

fn no_field_lowering(id: NodeId, name: &str, what: &str) -> EmitError {
    EmitError::at(
        id,
        format!("field `{name}` of {what} has no lowering: its producer has no record form here"),
    )
}

pub(super) use crate::refuse::record_in_tensor_position;
