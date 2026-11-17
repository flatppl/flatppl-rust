//! §09 `lu`: partial pivoting with the convention P A = L U.

use super::*;

impl Emitter<'_> {
    pub(super) fn lower_lu_record(
        &mut self,
        id: NodeId,
        field: &str,
    ) -> Option<Result<Lowered, EmitError>> {
        let Node::Call(call) = self.m.node(id) else {
            return None;
        };
        let (callee, args, broadcast) = match call.head {
            CallHead::User(callee) => (callee, call.args.to_vec(), false),
            CallHead::Builtin(head) if self.m.resolve(head) == "broadcast" => {
                let (&callee, args) = call.args.split_first()?;
                (callee, args.to_vec(), true)
            }
            _ => return None,
        };
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra" || member != "lu" {
            return None;
        }
        Some((|| {
            let [matrix] = args.as_slice() else {
                return Err(EmitError::at(id, "lu requires one square matrix"));
            };
            let a = self.lower_node(*matrix)?;
            let mut inputs = [self.typed_axes(*matrix, a)];
            let parent = if broadcast {
                Some(self.enter_broadcast(id, &mut inputs)?)
            } else {
                None
            };
            let (p, l, u) = self.matrix_lu(id, &inputs[0])?;
            let frame = self.broadcast_frame.clone();
            for (name, value) in [("P", p), ("L", l), ("U", u)] {
                let value = if let Some(parent) = &parent {
                    self.finish_broadcast(&value, &frame, parent.len())
                } else {
                    value
                };
                self.bind_field(id, name.to_owned(), Lowered::Tensor(value));
            }
            if let Some(parent) = parent {
                self.broadcast_frame = parent;
            }
            self.columns
                .get(&(self.column_container(id), field.to_owned()))
                .cloned()
                .ok_or_else(|| EmitError::at(id, format!("lu has no field `{field}`")))
        })())
    }

    fn matrix_lu(&mut self, id: NodeId, a: &Value) -> Result<(Value, Value, Value), EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(a) else {
            return Err(EmitError::at(id, "lu requires a static square matrix"));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(id, "lu requires a static square matrix"));
        };
        if n != m {
            return Err(EmitError::at(id, "lu requires a square matrix"));
        }
        let n = *n;
        let a = self.convert(a, ElemKind::Real);
        let axes = self.axes_of(&a);
        let batch = self.batch_rank(&a);
        let rows = self.axis_indices(&a, batch);
        let cols = self.axis_indices(&a, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let packed = self.matrix_loop(
            n,
            MATRIX_UNROLL_LIMIT,
            &[a, identity.clone()],
            |e, index, state| {
                let a = e.axes_view(&state[0], axes.clone());
                let p = e.axes_view(&state[1], axes.clone());
                let column = e.indexed_matrix_axis(&a, index, 1);
                let indices = e.axis_indices(&column, batch);
                let eligible = e.compare("GE", &indices, index);
                let magnitude = e.abs(&column);
                let infinity = e.inf_like(&column);
                let minus_inf = e.neg(&infinity);
                let candidates = e.select(&eligible, &magnitude, &minus_inf);
                let largest = e.reduce_max(&candidates);
                let candidates = e.compare("EQ", &candidates, &largest);
                let limit = e.int_value_const(n as i64);
                let candidates = e.select(&candidates, &indices, &limit);
                let pivot =
                    e.reduce_axis_lit("stablehlo.minimum", &n.to_string(), &candidates, batch);
                let current = e.compare("EQ", &rows, index);
                let pivot_rows = e.compare("EQ", &rows, &pivot);
                let a = e.swap_rows(&a, &current, &pivot_rows);
                let p = e.swap_rows(&p, &current, &pivot_rows);
                let column = e.indexed_matrix_axis(&a, index, 1);
                let row = e.indexed_matrix_axis(&a, index, 0);
                let at_pivot = e.compare("EQ", &indices, index);
                let zero = e.constant_like(0.0, &row);
                let selected = e.select(&at_pivot, &row, &zero);
                let pivot = e.reduce_sum_last_axis(&selected);
                let zero_scalar = e.scalar(0.0);
                let nonzero = e.compare("NE", &pivot, &zero_scalar);
                // A zero remaining column needs no elimination, even for a singular A.
                let one = e.scalar(1.0);
                let pivot = e.select(&nonzero, &pivot, &one);
                let below = e.compare("GT", &indices, index);
                let factors = e.div(&column, &pivot);
                let factors = e.select(&below, &factors, &zero);
                let factors = e.matrix_expand(&factors, &a, batch);
                let row = e.matrix_expand(&row, &a, batch + 1);
                let change = e.mul(&factors, &row);
                let reduced = e.sub(&a, &change);
                let trailing = e.compare("GT", &cols, index);
                let reduced = e.select(&trailing, &reduced, &a);
                let pivot_column = e.compare("EQ", &cols, index);
                let below = e.compare("GT", &rows, index);
                let store = e.and(&pivot_column, &below);
                vec![e.select(&store, &factors, &reduced), p]
            },
        );
        let packed_a = self.axes_view(&packed[0], axes.clone());
        let p = self.axes_view(&packed[1], axes);
        let zero = self.constant_like(0.0, &packed_a);
        let below = self.compare("GT", &rows, &cols);
        let lower = self.select(&below, &packed_a, &zero);
        let lower = self.add(&lower, &identity);
        let upper = self.select(&below, &zero, &packed_a);
        Ok((p, lower, upper))
    }
}
