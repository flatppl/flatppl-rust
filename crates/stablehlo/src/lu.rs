//! §09 `lu`: partial pivoting with the convention P A = L U.

use super::*;

struct LuFactors {
    permutation: Value,
    lower: (Value, Option<Value>),
    upper: (Value, Option<Value>),
}

impl Emitter<'_> {
    fn lu_argument(&self, id: NodeId) -> Option<(NodeId, bool)> {
        let Node::Call(call) = self.m.node(id) else {
            return None;
        };
        let (callee, args, broadcast) = match call.head {
            CallHead::User(callee) => (callee, call.args.as_ref(), false),
            CallHead::Builtin(head) if self.m.resolve(head) == "broadcast" => {
                let (&callee, args) = call.args.split_first()?;
                (callee, args, true)
            }
            _ => return None,
        };
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        let [matrix] = args else { return None };
        (module == "ext-linear-algebra" && member == "lu").then_some((*matrix, broadcast))
    }

    pub(super) fn lower_lu_record(
        &mut self,
        id: NodeId,
        field: &str,
    ) -> Option<Result<Lowered, EmitError>> {
        let (matrix, broadcast) = self.lu_argument(id)?;
        Some((|| {
            let complex = crate::complex::complex_leaf(self.type_of(matrix));
            if complex && field != "P" {
                return Err(EmitError::at(
                    id,
                    "complex LU factors require real/imaginary projections",
                ));
            }
            let factors = self.lower_lu_factors(id, matrix, broadcast)?;
            self.bind_field(id, "P".to_owned(), Lowered::Tensor(factors.permutation));
            if !complex {
                self.bind_field(id, "L".to_owned(), Lowered::Tensor(factors.lower.0));
                self.bind_field(id, "U".to_owned(), Lowered::Tensor(factors.upper.0));
            }
            self.columns
                .get(&(self.column_container(id), field.to_owned()))
                .cloned()
                .ok_or_else(|| EmitError::at(id, format!("lu has no field `{field}`")))
        })())
    }

    pub(crate) fn lower_lu_field_parts(
        &mut self,
        container: NodeId,
        field: &str,
    ) -> Option<Result<(Value, Value), EmitError>> {
        let id = self.resolve_tuple_projections(container);
        let (matrix, broadcast) = self.lu_argument(id)?;
        Some((|| {
            let factors = self.lower_lu_factors(id, matrix, broadcast)?;
            let (re, im) = match field {
                "P" => (factors.permutation, None),
                "L" => factors.lower,
                "U" => factors.upper,
                _ => return Err(EmitError::at(id, format!("lu has no field `{field}`"))),
            };
            let im = im.unwrap_or_else(|| self.constant_like(0.0, &re));
            Ok((re, im))
        })())
    }

    fn lower_lu_factors(
        &mut self,
        id: NodeId,
        matrix: NodeId,
        broadcast: bool,
    ) -> Result<LuFactors, EmitError> {
        let complex = crate::complex::complex_leaf(self.type_of(matrix));
        let mut inputs = if complex {
            let (re, im) = crate::complex::parts(self, matrix, &mut HashMap::new())?;
            vec![self.typed_axes(matrix, re), self.typed_axes(matrix, im)]
        } else {
            let value = self.lower_node(matrix)?;
            vec![self.typed_axes(matrix, value)]
        };
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut inputs)?)
        } else {
            None
        };
        let factors = self.matrix_lu(id, &inputs[0], inputs.get(1))?;
        let Some(parent) = parent else {
            return Ok(factors);
        };
        let frame = self.broadcast_frame.clone();
        let mut finish = |value: Value| self.finish_broadcast(&value, &frame, parent.len());
        let factors = LuFactors {
            permutation: finish(factors.permutation),
            lower: (finish(factors.lower.0), factors.lower.1.map(&mut finish)),
            upper: (finish(factors.upper.0), factors.upper.1.map(&mut finish)),
        };
        self.broadcast_frame = parent;
        Ok(factors)
    }

    fn lu_magnitude(&mut self, re: &Value, im: Option<&Value>) -> Value {
        let Some(im) = im else { return self.abs(re) };
        let ar = self.abs(re);
        let ai = self.abs(im);
        let scale = self.max(&ar, &ai);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let nonzero = self.compare("GT", &scale, &zero);
        let safe = self.select(&nonzero, &scale, &one);
        let re = self.div(re, &safe);
        let im = self.div(im, &safe);
        let rr = self.mul(&re, &re);
        let ii = self.mul(&im, &im);
        let norm = self.add(&rr, &ii);
        let norm = self.select(&nonzero, &norm, &one);
        let norm = self.sqrt(&norm);
        self.mul(&scale, &norm)
    }

    fn matrix_lu(
        &mut self,
        id: NodeId,
        a: &Value,
        imaginary: Option<&Value>,
    ) -> Result<LuFactors, EmitError> {
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
        let mut initial = vec![a, identity.clone()];
        if let Some(im) = imaginary {
            initial.push(self.convert(im, ElemKind::Real));
        }
        let packed = self.matrix_loop(n, MATRIX_UNROLL_LIMIT, &initial, |e, index, state| {
            let a = e.axes_view(&state[0], axes.clone());
            let p = e.axes_view(&state[1], axes.clone());
            let ai = state.get(2).map(|v| e.axes_view(v, axes.clone()));
            let column = e.indexed_matrix_axis(&a, index, 1);
            let column_i = ai.as_ref().map(|v| e.indexed_matrix_axis(v, index, 1));
            let indices = e.axis_indices(&column, batch);
            let eligible = e.compare("GE", &indices, index);
            let magnitude = e.lu_magnitude(&column, column_i.as_ref());
            let infinity = e.inf_like(&column);
            let minus_inf = e.neg(&infinity);
            let candidates = e.select(&eligible, &magnitude, &minus_inf);
            let largest = e.reduce_max(&candidates);
            let candidates = e.compare("EQ", &candidates, &largest);
            let limit = e.int_value_const(n as i64);
            let candidates = e.select(&candidates, &indices, &limit);
            let pivot = e.reduce_axis_lit("stablehlo.minimum", &n.to_string(), &candidates, batch);
            let current = e.compare("EQ", &rows, index);
            let pivot_rows = e.compare("EQ", &rows, &pivot);
            let a = e.swap_rows(&a, &current, &pivot_rows);
            let ai = ai.map(|v| e.swap_rows(&v, &current, &pivot_rows));
            let p = e.swap_rows(&p, &current, &pivot_rows);
            let column = e.indexed_matrix_axis(&a, index, 1);
            let row = e.indexed_matrix_axis(&a, index, 0);
            let at_pivot = e.compare("EQ", &indices, index);
            let zero = e.constant_like(0.0, &row);
            let selected = e.select(&at_pivot, &row, &zero);
            let pivot = e.reduce_sum_last_axis(&selected);
            let zero_scalar = e.scalar(0.0);
            let one = e.scalar(1.0);
            let below = e.compare("GT", &indices, index);
            let trailing = e.compare("GT", &cols, index);
            let pivot_column = e.compare("EQ", &cols, index);
            let below_rows = e.compare("GT", &rows, index);
            let store = e.and(&pivot_column, &below_rows);
            // Preserve prior packed L columns, update the trailing block, then
            // store multipliers. A zero remaining column needs no elimination.
            let update = |e: &mut Self, matrix: &Value, factors: &Value, change: &Value| {
                let reduced = e.sub(matrix, change);
                let reduced = e.select(&trailing, &reduced, matrix);
                e.select(&store, factors, &reduced)
            };
            if let Some(ai) = ai {
                let column_i = e.indexed_matrix_axis(&ai, index, 1);
                let row_i = e.indexed_matrix_axis(&ai, index, 0);
                let selected = e.select(&at_pivot, &row_i, &zero);
                let pivot_i = e.reduce_sum_last_axis(&selected);
                // Normalize the divisor before squaring. Finite matrices at
                // scales 1e±150 must not overflow or erase their multipliers.
                let pr = e.abs(&pivot);
                let pi = e.abs(&pivot_i);
                let scale = e.max(&pr, &pi);
                let nonzero = e.compare("NE", &scale, &zero_scalar);
                let scale = e.select(&nonzero, &scale, &one);
                let pr = e.div(&pivot, &scale);
                let pi = e.div(&pivot_i, &scale);
                let rr = e.mul(&pr, &pr);
                let ii = e.mul(&pi, &pi);
                let denominator = e.add(&rr, &ii);
                let denominator = e.select(&nonzero, &denominator, &one);
                let cr = e.div(&column, &scale);
                let ci = e.div(&column_i, &scale);
                let rr = e.mul(&cr, &pr);
                let ii = e.mul(&ci, &pi);
                let ir = e.mul(&ci, &pr);
                let ri = e.mul(&cr, &pi);
                let fr = e.add(&rr, &ii);
                let fi = e.sub(&ir, &ri);
                let fr = e.div(&fr, &denominator);
                let fi = e.div(&fi, &denominator);
                let fr = e.select(&below, &fr, &zero);
                let fi = e.select(&below, &fi, &zero);
                let fr = e.matrix_expand(&fr, &a, batch);
                let fi = e.matrix_expand(&fi, &a, batch);
                let row = e.matrix_expand(&row, &a, batch + 1);
                let row_i = e.matrix_expand(&row_i, &ai, batch + 1);
                let rr = e.mul(&fr, &row);
                let ii = e.mul(&fi, &row_i);
                let ir = e.mul(&fi, &row);
                let ri = e.mul(&fr, &row_i);
                let dr = e.sub(&rr, &ii);
                let di = e.add(&ir, &ri);
                vec![update(e, &a, &fr, &dr), p, update(e, &ai, &fi, &di)]
            } else {
                let nonzero = e.compare("NE", &pivot, &zero_scalar);
                let pivot = e.select(&nonzero, &pivot, &one);
                let factors = e.div(&column, &pivot);
                let factors = e.select(&below, &factors, &zero);
                let factors = e.matrix_expand(&factors, &a, batch);
                let row = e.matrix_expand(&row, &a, batch + 1);
                let change = e.mul(&factors, &row);
                vec![update(e, &a, &factors, &change), p]
            }
        });
        let a = self.axes_view(&packed[0], axes.clone());
        let p = self.axes_view(&packed[1], axes.clone());
        let zero = self.constant_like(0.0, &a);
        let below = self.compare("GT", &rows, &cols);
        let lower = self.select(&below, &a, &zero);
        let lower = self.add(&lower, &identity);
        let upper = self.select(&below, &zero, &a);
        let imaginary = packed.get(2).map(|v| {
            let value = self.axes_view(v, axes);
            (
                self.select(&below, &value, &zero),
                self.select(&below, &zero, &value),
            )
        });
        Ok(LuFactors {
            permutation: p,
            lower: (lower, imaginary.as_ref().map(|(lower, _)| lower.clone())),
            upper: (upper, imaginary.map(|(_, upper)| upper)),
        })
    }
}
