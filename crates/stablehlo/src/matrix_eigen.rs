//! General complex Schur iteration and right eigenvectors, spec §09.
//! Full-row deflation preserves similarity without assuming Hessenberg input.

use super::*;

type Pair = (Value, Value);

struct Eigen {
    values: Pair,
    vectors: Pair,
    real_spectrum: Value,
}

impl Emitter<'_> {
    fn eigen_call(&self, id: NodeId) -> Option<(Vec<NodeId>, bool)> {
        let id = self.resolve_tuple_projections(id);
        let Node::Call(call) = self.node(id) else {
            return None;
        };
        let (callee, args, broadcast) = match call.head {
            CallHead::User(callee) => (callee, call.args.to_vec(), false),
            CallHead::Builtin(head) if self.resolve(head) == "broadcast" => {
                let (&callee, args) = call.args.split_first()?;
                (callee, args.to_vec(), true)
            }
            _ => return None,
        };
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        (module == "ext-linear-algebra" && member == "eigen").then_some((args, broadcast))
    }

    pub(super) fn lower_eigen_record(
        &mut self,
        id: NodeId,
        _field: &str,
    ) -> Option<Result<Lowered, EmitError>> {
        self.eigen_call(id)?;
        Some(Err(EmitError::at(
            id,
            "complex eigenvalues and eigenvectors require real/imaginary projections",
        )))
    }

    pub(crate) fn lower_eigen_field_parts(
        &mut self,
        container: NodeId,
        field: &str,
    ) -> Option<Result<Pair, EmitError>> {
        let (args, broadcast) = self.eigen_call(container)?;
        Some((|| {
            let result = self.emit_eigen(container, &args, broadcast)?;
            match field {
                "values" => Ok(result.values),
                "vectors" => Ok(result.vectors),
                _ => Err(EmitError::at(
                    container,
                    "eigen has fields values and vectors",
                )),
            }
        })())
    }

    pub(super) fn lower_eigen_extremum(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra" || !matches!(member.as_str(), "eigmax" | "eigmin") {
            return None;
        }
        Some((|| {
            let result = self.emit_eigen(id, args, broadcast)?;
            self.record_runtime_error(&result.real_spectrum, 11);
            let mut values = result.values.0;
            if broadcast {
                let axes = self.axes_of(&values);
                let outer = axes.layers[..axes.layers.len() - 1].iter().sum::<usize>();
                values = self.axes_view(
                    &values,
                    Axes {
                        batch: axes.batch + outer,
                        layers: vec![1],
                    },
                );
            }
            let value = if member == "eigmax" {
                self.reduce_max(&values)
            } else {
                self.reduce_min(&values)
            };
            let value = if broadcast {
                let mut axes = self.axes_of(&result.values.1);
                axes.layers.pop();
                self.axes_view(&value, axes)
            } else {
                value
            };
            let nan = self.scalar(f64::NAN);
            Ok(self.select(&result.real_spectrum, &value, &nan))
        })())
    }

    fn emit_eigen(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Eigen, EmitError> {
        let [arg] = args else {
            return Err(EmitError::at(id, "eigen requires one square matrix"));
        };
        let (re, im) = crate::complex::parts(self, *arg, &mut HashMap::new())?;
        let (re, im) = self.broadcast_pair(&re, &im);
        let mut values = [self.typed_axes(*arg, re), self.typed_axes(*arg, im)];
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let mut result = self.matrix_eigen(id, (values[0].clone(), values[1].clone()))?;
        if let Some(parent) = parent {
            let frame = self.broadcast_frame.clone();
            result.values.0 = self.finish_broadcast(&result.values.0, &frame, parent.len());
            result.values.1 = self.finish_broadcast(&result.values.1, &frame, parent.len());
            result.vectors.0 = self.finish_broadcast(&result.vectors.0, &frame, parent.len());
            result.vectors.1 = self.finish_broadcast(&result.vectors.1, &frame, parent.len());
            result.real_spectrum =
                self.finish_broadcast(&result.real_spectrum, &frame, parent.len());
            self.broadcast_frame = parent;
        }
        Ok(result)
    }

    fn eig_transpose(&mut self, value: &Value) -> Value {
        let batch = self.batch_rank(value);
        let mut permutation = (0..batch as u64).collect::<Vec<_>>();
        permutation.extend([batch as u64 + 1, batch as u64]);
        let result = self.transpose(value, &permutation);
        self.axes_view(&result, self.axes_of(value))
    }

    fn eig_matmul(&mut self, a: &Pair, b: &Pair) -> Pair {
        let rr = self.matmat(&a.0, &b.0);
        let ii = self.matmat(&a.1, &b.1);
        let ri = self.matmat(&a.0, &b.1);
        let ir = self.matmat(&a.1, &b.0);
        (self.sub(&rr, &ii), self.add(&ri, &ir))
    }

    fn eig_axis(&mut self, matrix: &Value, index: &Value, axis: usize) -> Value {
        let axis = self.batch_rank(matrix) + axis;
        let indices = self.axis_indices(matrix, axis);
        let selected = self.compare("EQ", &indices, index);
        let zero = self.constant_like(0.0, matrix);
        let selected = self.select(&selected, matrix, &zero);
        self.reduce_axis("stablehlo.add", "0.000000e+00", &selected, axis)
    }

    fn eig_entry(&mut self, matrix: &Value, row: &Value, col: &Value) -> Value {
        let batch = self.batch_rank(matrix);
        let rows = self.axis_indices(matrix, batch);
        let cols = self.axis_indices(matrix, batch + 1);
        let r = self.compare("EQ", &rows, row);
        let c = self.compare("EQ", &cols, col);
        let selected = self.and(&r, &c);
        let zero = self.constant_like(0.0, matrix);
        let selected = self.select(&selected, matrix, &zero);
        self.reduce_sum(&selected)
    }

    fn eig_sqrt(&mut self, value: &Value) -> Value {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", value, &zero);
        let safe = self.select(&positive, value, &one);
        let root = self.sqrt(&safe);
        self.select(&positive, &root, &zero)
    }

    fn eig_hypot(&mut self, re: &Value, im: &Value) -> Value {
        let ar = self.abs(re);
        let ai = self.abs(im);
        let scale = self.max(&ar, &ai);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let safe = self.select(&positive, &scale, &one);
        let re = self.div(re, &safe);
        let im = self.div(im, &safe);
        let rr = self.mul(&re, &re);
        let ii = self.mul(&im, &im);
        let square = self.add(&rr, &ii);
        let root = self.eig_sqrt(&square);
        self.mul(&scale, &root)
    }

    fn eig_normalize(&mut self, vector: &Pair) -> Pair {
        let ar = self.abs(&vector.0);
        let ai = self.abs(&vector.1);
        let magnitude = self.max(&ar, &ai);
        let scale = self.reduce_max(&magnitude);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let scale = self.select(&positive, &scale, &one);
        let re = self.div(&vector.0, &scale);
        let im = self.div(&vector.1, &scale);
        let rr = self.mul(&re, &re);
        let ii = self.mul(&im, &im);
        let norm2 = self.add(&rr, &ii);
        let norm2 = self.reduce_sum(&norm2);
        let norm = self.eig_sqrt(&norm2);
        let norm = self.select(&positive, &norm, &one);
        (self.div(&re, &norm), self.div(&im, &norm))
    }

    /// Fix the arbitrary phase with a largest, positive-real coordinate. This
    /// removes the QR iteration's phase from derivatives of simple eigenvectors.
    fn eig_canonical_vector(&mut self, vector: &Pair) -> Pair {
        let vector = self.eig_normalize(vector);
        let magnitude = self.eig_hypot(&vector.0, &vector.1);
        let largest = self.reduce_max(&magnitude);
        let at_largest = self.compare("EQ", &magnitude, &largest);
        let rows = self.axis_indices(&vector.0, self.batch_rank(&vector.0));
        let limit = self.int_value_const(i32::MAX.into());
        let candidates = self.select(&at_largest, &rows, &limit);
        let row = self.reduce_axis_lit(
            "stablehlo.minimum",
            &i32::MAX.to_string(),
            &candidates,
            self.batch_rank(&vector.0),
        );
        let selected = self.compare("EQ", &rows, &row);
        let zero = self.constant_like(0.0, &vector.0);
        let re = self.select(&selected, &vector.0, &zero);
        let im = self.select(&selected, &vector.1, &zero);
        let re = self.reduce_sum(&re);
        let im = self.reduce_sum(&im);
        let norm = self.eig_hypot(&re, &im);
        let one = self.scalar(1.0);
        let zero = self.scalar(0.0);
        let nonzero = self.compare("GT", &norm, &zero);
        let norm = self.select(&nonzero, &norm, &one);
        let pr = self.div(&re, &norm);
        let pi = self.div(&im, &norm);
        let rr = self.mul(&vector.0, &pr);
        let ii = self.mul(&vector.1, &pi);
        let ir = self.mul(&vector.1, &pr);
        let ri = self.mul(&vector.0, &pi);
        (self.add(&rr, &ii), self.sub(&ir, &ri))
    }

    /// Unit-normalized Householder vectors avoid column-scale overflow.
    fn eig_qr(&mut self, input: &Pair, active: &Value, n: u64) -> Pair {
        let batch = self.batch_rank(&input.0);
        let axes = self.axes_of(&input.0);
        let rows = self.axis_indices(&input.0, batch);
        let cols = self.axis_indices(&input.0, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let zero_matrix = self.constant_like(0.0, &input.0);
        let state = self.matrix_loop(
            n,
            0,
            &[input.0.clone(), input.1.clone(), identity, zero_matrix],
            |e, k, state| {
                let state = state
                    .iter()
                    .map(|v| e.axes_view(v, axes.clone()))
                    .collect::<Vec<_>>();
                let r = (state[0].clone(), state[1].clone());
                let q = (state[2].clone(), state[3].clone());
                let cr = e.eig_axis(&r.0, k, 1);
                let ci = e.eig_axis(&r.1, k, 1);
                let indices = e.axis_indices(&cr, batch);
                let below = e.compare("GE", &indices, k);
                let inside = e.compare("LT", &indices, active);
                let eligible = e.and(&below, &inside);
                let zero = e.constant_like(0.0, &cr);
                let cr = e.select(&eligible, &cr, &zero);
                let ci = e.select(&eligible, &ci, &zero);
                let ar = e.abs(&cr);
                let ai = e.abs(&ci);
                let magnitude = e.max(&ar, &ai);
                let scale = e.reduce_max(&magnitude);
                let one = e.scalar(1.0);
                let zero_scalar = e.scalar(0.0);
                let positive = e.compare("GT", &scale, &zero_scalar);
                let scale = e.select(&positive, &scale, &one);
                let vr = e.div(&cr, &scale);
                let vi = e.div(&ci, &scale);
                let rr = e.mul(&vr, &vr);
                let ii = e.mul(&vi, &vi);
                let square = e.add(&rr, &ii);
                let square = e.reduce_sum(&square);
                let norm = e.eig_sqrt(&square);
                let at_k = e.compare("EQ", &indices, k);
                let first_r = e.select(&at_k, &vr, &zero);
                let first_i = e.select(&at_k, &vi, &zero);
                let first_r = e.reduce_sum(&first_r);
                let first_i = e.reduce_sum(&first_i);
                let first = e.eig_hypot(&first_r, &first_i);
                let nonzero = e.compare("GT", &first, &zero_scalar);
                let safe = e.select(&nonzero, &first, &one);
                let phase_r = e.div(&first_r, &safe);
                let phase_i = e.div(&first_i, &safe);
                let phase_r = e.select(&nonzero, &phase_r, &one);
                let phase_r = e.mul(&phase_r, &norm);
                let phase_i = e.mul(&phase_i, &norm);
                let add_r = e.select(&at_k, &phase_r, &zero);
                let add_i = e.select(&at_k, &phase_i, &zero);
                let vr = e.add(&vr, &add_r);
                let vi = e.add(&vi, &add_i);
                let (vr, vi) = e.eig_normalize(&(vr, vi));
                let two = e.scalar(2.0);
                let beta = e.select(&positive, &two, &zero_scalar);
                let vr_rows = e.matrix_expand(&vr, &r.0, batch);
                let vi_rows = e.matrix_expand(&vi, &r.1, batch);
                let rr = e.mul(&vr_rows, &r.0);
                let ii = e.mul(&vi_rows, &r.1);
                let ri = e.mul(&vr_rows, &r.1);
                let ir = e.mul(&vi_rows, &r.0);
                let dr = e.add(&rr, &ii);
                let di = e.sub(&ri, &ir);
                let dr = e.reduce_axis("stablehlo.add", "0.000000e+00", &dr, batch);
                let di = e.reduce_axis("stablehlo.add", "0.000000e+00", &di, batch);
                let dr = e.mul(&dr, &beta);
                let di = e.mul(&di, &beta);
                let dr_cols = e.matrix_expand(&dr, &r.0, batch + 1);
                let di_cols = e.matrix_expand(&di, &r.1, batch + 1);
                let rr = e.mul(&vr_rows, &dr_cols);
                let ii = e.mul(&vi_rows, &di_cols);
                let ri = e.mul(&vr_rows, &di_cols);
                let ir = e.mul(&vi_rows, &dr_cols);
                let change_r = e.sub(&rr, &ii);
                let change_i = e.add(&ri, &ir);
                let r = (e.sub(&r.0, &change_r), e.sub(&r.1, &change_i));
                let rr = e.matvec(&q.0, &vr);
                let ii = e.matvec(&q.1, &vi);
                let ri = e.matvec(&q.0, &vi);
                let ir = e.matvec(&q.1, &vr);
                let pr = e.sub(&rr, &ii);
                let pi = e.add(&ri, &ir);
                let pr = e.mul(&pr, &beta);
                let pi = e.mul(&pi, &beta);
                let pr_rows = e.matrix_expand(&pr, &q.0, batch);
                let pi_rows = e.matrix_expand(&pi, &q.1, batch);
                let vr_cols = e.matrix_expand(&vr, &q.0, batch + 1);
                let vi_cols = e.matrix_expand(&vi, &q.1, batch + 1);
                let rr = e.mul(&pr_rows, &vr_cols);
                let ii = e.mul(&pi_rows, &vi_cols);
                let ir = e.mul(&pi_rows, &vr_cols);
                let ri = e.mul(&pr_rows, &vi_cols);
                let change_r = e.add(&rr, &ii);
                let change_i = e.sub(&ir, &ri);
                vec![r.0, r.1, e.sub(&q.0, &change_r), e.sub(&q.1, &change_i)]
            },
        );
        (
            self.axes_view(&state[2], axes.clone()),
            self.axes_view(&state[3], axes),
        )
    }

    fn eig_complex_sqrt(&mut self, re: &Value, im: &Value) -> Pair {
        let magnitude = self.eig_hypot(re, im);
        let half = self.scalar(0.5);
        let plus = self.add(&magnitude, re);
        let minus = self.sub(&magnitude, re);
        let plus = self.mul(&plus, &half);
        let minus = self.mul(&minus, &half);
        let r = self.eig_sqrt(&plus);
        let i = self.eig_sqrt(&minus);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let two = self.scalar(2.0);
        let r_nonzero = self.compare("GT", &r, &zero);
        let i_nonzero = self.compare("GT", &i, &zero);
        let safe_r = self.select(&r_nonzero, &r, &one);
        let safe_i = self.select(&i_nonzero, &i, &one);
        let safe_r = self.mul(&safe_r, &two);
        let safe_i = self.mul(&safe_i, &two);
        let right_i = self.div(im, &safe_r);
        let left_r = self.div(im, &safe_i);
        let left_r = self.abs(&left_r);
        let negative_im = self.compare("LT", im, &zero);
        let negative_i = self.neg(&i);
        let left_i = self.select(&negative_im, &negative_i, &i);
        let positive_re = self.compare("GE", re, &zero);
        (
            self.select(&positive_re, &r, &left_r),
            self.select(&positive_re, &right_i, &left_i),
        )
    }

    fn eig_shift(&mut self, t: &Pair, last: &Value) -> Pair {
        let one = self.int_value_const(1);
        let previous = self.sub(last, &one);
        let entries = [
            (&previous, &previous),
            (&previous, last),
            (last, &previous),
            (last, last),
        ]
        .map(|(r, c)| (self.eig_entry(&t.0, r, c), self.eig_entry(&t.1, r, c)));
        let mut scale = self.scalar(0.0);
        for value in &entries {
            let re = self.abs(&value.0);
            let im = self.abs(&value.1);
            let largest = self.max(&re, &im);
            scale = self.max(&scale, &largest);
        }
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let safe = self.select(&positive, &scale, &one);
        let [a, b, c, d] = entries.map(|v| (self.div(&v.0, &safe), self.div(&v.1, &safe)));
        let half = self.scalar(0.5);
        let two = self.scalar(2.0);
        let xr = self.sub(&a.0, &d.0);
        let xi = self.sub(&a.1, &d.1);
        let xr = self.mul(&xr, &half);
        let xi = self.mul(&xi, &half);
        let rr = self.mul(&xr, &xr);
        let ii = self.mul(&xi, &xi);
        let brcr = self.mul(&b.0, &c.0);
        let bici = self.mul(&b.1, &c.1);
        let disc_r = self.sub(&rr, &ii);
        let product_r = self.sub(&brcr, &bici);
        let disc_r = self.add(&disc_r, &product_r);
        let ri = self.mul(&xr, &xi);
        let ri = self.mul(&ri, &two);
        let brci = self.mul(&b.0, &c.1);
        let bicr = self.mul(&b.1, &c.0);
        let product_i = self.add(&brci, &bicr);
        let disc_i = self.add(&ri, &product_i);
        let root = self.eig_complex_sqrt(&disc_r, &disc_i);
        let mr = self.add(&a.0, &d.0);
        let mi = self.add(&a.1, &d.1);
        let mr = self.mul(&mr, &half);
        let mi = self.mul(&mi, &half);
        let plus = (self.add(&mr, &root.0), self.add(&mi, &root.1));
        let minus = (self.sub(&mr, &root.0), self.sub(&mi, &root.1));
        let pr = self.sub(&plus.0, &d.0);
        let pi = self.sub(&plus.1, &d.1);
        let nr = self.sub(&minus.0, &d.0);
        let ni = self.sub(&minus.1, &d.1);
        let p = self.eig_hypot(&pr, &pi);
        let q = self.eig_hypot(&nr, &ni);
        let closer = self.compare("LE", &p, &q);
        let re = self.select(&closer, &plus.0, &minus.0);
        let im = self.select(&closer, &plus.1, &minus.1);
        (self.mul(&re, &scale), self.mul(&im, &scale))
    }

    fn eig_divide(&mut self, numerator: &Pair, denominator: &Pair) -> Pair {
        let ar = self.abs(&denominator.0);
        let ai = self.abs(&denominator.1);
        let scale = self.max(&ar, &ai);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let nonzero = self.compare("GT", &scale, &zero);
        let scale = self.select(&nonzero, &scale, &one);
        let dr = self.div(&denominator.0, &scale);
        let di = self.div(&denominator.1, &scale);
        let rr = self.mul(&dr, &dr);
        let ii = self.mul(&di, &di);
        let norm = self.add(&rr, &ii);
        let norm = self.select(&nonzero, &norm, &one);
        let nr = self.div(&numerator.0, &scale);
        let ni = self.div(&numerator.1, &scale);
        let rr = self.mul(&nr, &dr);
        let ii = self.mul(&ni, &di);
        let ir = self.mul(&ni, &dr);
        let ri = self.mul(&nr, &di);
        let re = self.add(&rr, &ii);
        let im = self.sub(&ir, &ri);
        (self.div(&re, &norm), self.div(&im, &norm))
    }

    fn eig_vectors(&mut self, t: &Pair, z: &Pair, n: u64, epsilon: f64) -> Pair {
        let axes = self.axes_of(&t.0);
        let batch = self.batch_rank(&t.0);
        let zero_matrix = self.constant_like(0.0, &t.0);
        let zero_vector = self.reduce_sum_last_axis(&zero_matrix);
        let indices = self.axis_indices(&zero_vector, batch);
        let columns = self.axis_indices(&t.0, batch + 1);
        let initial = [zero_matrix.clone(), zero_matrix];
        let result = self.matrix_loop(n, 0, &initial, |e, j, state| {
            let lambda = (e.eig_entry(&t.0, j, j), e.eig_entry(&t.1, j, j));
            let basis = e.compare("EQ", &indices, j);
            let basis = e.convert(&basis, ElemKind::Real);
            let w = e.matrix_loop(n, 0, &[basis, zero_vector.clone()], |e, step, w| {
                let w = (
                    e.axes_view(&w[0], e.axes_of(&zero_vector)),
                    e.axes_view(&w[1], e.axes_of(&zero_vector)),
                );
                let last = e.int_value_const(n as i64 - 1);
                let i = e.sub(&last, step);
                let row = (e.eig_axis(&t.0, &i, 0), e.eig_axis(&t.1, &i, 0));
                let after = e.compare("GT", &indices, &i);
                let before = e.compare("LE", &indices, j);
                let selected = e.and(&after, &before);
                let rr = e.mul(&row.0, &w.0);
                let ii = e.mul(&row.1, &w.1);
                let ri = e.mul(&row.0, &w.1);
                let ir = e.mul(&row.1, &w.0);
                let rhs_r = e.sub(&rr, &ii);
                let rhs_i = e.add(&ri, &ir);
                let rhs_r = e.select(&selected, &rhs_r, &zero_vector);
                let rhs_i = e.select(&selected, &rhs_i, &zero_vector);
                let rhs_r = e.reduce_sum(&rhs_r);
                let rhs_i = e.reduce_sum(&rhs_i);
                let rhs = (e.neg(&rhs_r), e.neg(&rhs_i));
                let diagonal = (e.eig_entry(&t.0, &i, &i), e.eig_entry(&t.1, &i, &i));
                let dr = e.sub(&diagonal.0, &lambda.0);
                let di = e.sub(&diagonal.1, &lambda.1);
                let magnitude = e.eig_hypot(&dr, &di);
                let row_magnitude = e.eig_hypot(&row.0, &row.1);
                let row_scale = e.reduce_max(&row_magnitude);
                let lambda_scale = e.eig_hypot(&lambda.0, &lambda.1);
                let row_scale = e.max(&row_scale, &lambda_scale);
                let coefficient = e.scalar(8.0 * epsilon);
                let minimum = e.mul(&row_scale, &coefficient);
                let tiny = e.scalar(match e.dtype {
                    Dtype::F32 => f32::MIN_POSITIVE as f64,
                    Dtype::F64 => f64::MIN_POSITIVE,
                });
                let minimum = e.max(&minimum, &tiny);
                let small = e.compare("LT", &magnitude, &minimum);
                let zero = e.scalar(0.0);
                let one = e.scalar(1.0);
                let nonzero = e.compare("GT", &magnitude, &zero);
                let safe = e.select(&nonzero, &magnitude, &one);
                let ratio = e.div(&minimum, &safe);
                let adjusted_r = e.mul(&dr, &ratio);
                let adjusted_i = e.mul(&di, &ratio);
                let adjusted_r = e.select(&nonzero, &adjusted_r, &minimum);
                let dr = e.select(&small, &adjusted_r, &dr);
                let di = e.select(&small, &adjusted_i, &di);
                let solution = e.eig_divide(&rhs, &(dr, di));
                let at_i = e.compare("EQ", &indices, &i);
                let active = e.compare("LT", &i, j);
                let replace = e.and(&at_i, &active);
                let wr = e.select(&replace, &solution.0, &w.0);
                let wi = e.select(&replace, &solution.1, &w.1);
                let normalized = e.eig_normalize(&(wr, wi));
                vec![normalized.0, normalized.1]
            });
            let wr = e.axes_view(&w[0], e.axes_of(&zero_vector));
            let wi = e.axes_view(&w[1], e.axes_of(&zero_vector));
            let rr = e.matvec(&z.0, &wr);
            let ii = e.matvec(&z.1, &wi);
            let ri = e.matvec(&z.0, &wi);
            let ir = e.matvec(&z.1, &wr);
            let vr = e.sub(&rr, &ii);
            let vi = e.add(&ri, &ir);
            let vector = e.eig_canonical_vector(&(vr, vi));
            let old_r = e.axes_view(&state[0], axes.clone());
            let old_i = e.axes_view(&state[1], axes.clone());
            let vr = e.matrix_expand(&vector.0, &old_r, batch);
            let vi = e.matrix_expand(&vector.1, &old_i, batch);
            let selected = e.compare("EQ", &columns, j);
            vec![
                e.select(&selected, &vr, &old_r),
                e.select(&selected, &vi, &old_i),
            ]
        });
        (
            self.axes_view(&result[0], axes.clone()),
            self.axes_view(&result[1], axes),
        )
    }

    fn matrix_eigen(&mut self, id: NodeId, input: Pair) -> Result<Eigen, EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(&input.0) else {
            return Err(EmitError::at(id, "eigen requires a static square matrix"));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(id, "eigen requires static matrix dimensions"));
        };
        if n != m || *n > i32::MAX as u64 {
            return Err(EmitError::at(
                id,
                "eigen requires a square matrix with supported dimensions",
            ));
        }
        let n = *n;
        let axes = self.axes_of(&input.0);
        let batch = self.batch_rank(&input.0);
        let input = (
            self.convert(&input.0, ElemKind::Real),
            self.convert(&input.1, ElemKind::Real),
        );
        let zero_matrix = self.constant_like(0.0, &input.0);
        if n == 0 {
            let mut dims = shape(&input.0.ty)[..batch].to_vec();
            dims.push(Some(0));
            let values = self.constant(0.0, MlirTy::Ranked(dims));
            let values = self.axes_view(
                &values,
                Axes {
                    batch,
                    layers: vec![1],
                },
            );
            return Ok(Eigen {
                values: (values.clone(), values),
                vectors: (zero_matrix.clone(), zero_matrix),
                real_spectrum: self.bool_value_const(false),
            });
        }
        let epsilon = match self.dtype {
            Dtype::F32 => f32::EPSILON as f64,
            Dtype::F64 => f64::EPSILON,
        };
        let ar = self.abs(&input.0);
        let ai = self.abs(&input.1);
        let magnitude = self.max(&ar, &ai);
        let scale = self.reduce_max(&magnitude);
        let infinity = self.inf_like(&scale);
        let finite = self.compare("LT", &scale, &infinity);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let valid_scale = self.and(&positive, &finite);
        let safe_scale = self.select(&valid_scale, &scale, &one);
        let tr = self.div(&input.0, &safe_scale);
        let ti = self.div(&input.1, &safe_scale);
        let tr = self.select(&finite, &tr, &zero_matrix);
        let ti = self.select(&finite, &ti, &zero_matrix);
        let normalized_input = (tr.clone(), ti.clone());
        let rows = self.axis_indices(&tr, batch);
        let cols = self.axis_indices(&tr, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let scalar = self.reduce_sum(&zero_matrix);
        let active = self.constant_like(n as f64, &scalar);
        let active = self.convert(&active, ElemKind::Int);
        let counter = self.constant_like(0.0, &scalar);
        let counter = self.convert(&counter, ElemKind::Int);
        let scalar_axes = self.axes_of(&scalar);
        let initial = [tr, ti, identity, zero_matrix.clone(), active, counter];
        let limit = n
            .checked_mul(256)
            .ok_or_else(|| EmitError::at(id, "eigen iteration bound overflow"))?;
        let state = self.matrix_loop(limit, 0, &initial, |e, _, state| {
            let state = state
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    e.axes_view(
                        v,
                        if i < 4 {
                            axes.clone()
                        } else {
                            scalar_axes.clone()
                        },
                    )
                })
                .collect::<Vec<_>>();
            let one_i = e.int_value_const(1);
            let unfinished = e.compare("GT", &state[4], &one_i);
            e.matrix_if_values(&unfinished, &state, |e| {
                let mut state = state.clone();
                let last = e.sub(&state[4], &one_i);
                let prior = e.sub(&last, &one_i);
                let last_r = e.eig_entry(&state[0], &last, &last);
                let last_i = e.eig_entry(&state[1], &last, &last);
                let prior_r = e.eig_entry(&state[0], &prior, &prior);
                let prior_i = e.eig_entry(&state[1], &prior, &prior);
                let last_size = e.eig_hypot(&last_r, &last_i);
                let prior_size = e.eig_hypot(&prior_r, &prior_i);
                let local_scale = e.max(&last_size, &prior_size);
                let at_row = e.compare("EQ", &rows, &last);
                let before = e.compare("LT", &cols, &last);
                let off_diagonal = e.and(&at_row, &before);
                let magnitude = e.eig_hypot(&state[0], &state[1]);
                let off = e.select(&off_diagonal, &magnitude, &zero_matrix);
                let off = e.reduce_max(&off);
                let tolerance = e.scalar(64.0 * epsilon);
                let tolerance = e.mul(&tolerance, &local_scale);
                let small = e.compare("LE", &off, &tolerance);
                let unfinished = e.compare("GT", &state[4], &one_i);
                let deflate = e.and(&small, &unfinished);
                let keep_going = e.not(&small);
                let qr_step = e.and(&keep_going, &unfinished);
                let clear = e.and(&deflate, &off_diagonal);
                state[0] = e.select(&clear, &zero_matrix, &state[0]);
                state[1] = e.select(&clear, &zero_matrix, &state[1]);
                state[4] = e.select(&deflate, &last, &state[4]);
                let next_counter = e.add(&state[5], &one_i);
                let zero_i = e.int_value_const(0);
                let counter = e.select(&deflate, &zero_i, &state[5]);
                state[5] = e.select(&qr_step, &next_counter, &counter);
                e.matrix_if_values(&qr_step, &state, |e| {
                    let t = (state[0].clone(), state[1].clone());
                    let z = (state[2].clone(), state[3].clone());
                    let mu = e.eig_shift(&t, &last);
                    let period = e.int_value_const(32);
                    let remainder = e.rem(&state[5], &period);
                    let exceptional = e.compare("EQ", &remainder, &zero_i);
                    let three_quarters = e.scalar(0.75);
                    let quarter = e.scalar(0.25);
                    let escape_r = e.mul(&off, &three_quarters);
                    let escape_i = e.mul(&off, &quarter);
                    let escape_r = e.add(&last_r, &escape_r);
                    let escape_i = e.add(&last_i, &escape_i);
                    let mu_r = e.select(&exceptional, &escape_r, &mu.0);
                    let mu_i = e.select(&exceptional, &escape_i, &mu.1);
                    let shift_r = e.select(&diagonal, &mu_r, &zero_matrix);
                    let shift_i = e.select(&diagonal, &mu_i, &zero_matrix);
                    let shifted_r = e.sub(&t.0, &shift_r);
                    let shifted_i = e.sub(&t.1, &shift_i);
                    let inside_r = e.compare("LT", &rows, &state[4]);
                    let inside_c = e.compare("LT", &cols, &state[4]);
                    let inside = e.and(&inside_r, &inside_c);
                    let inside = e.and(&inside, &qr_step);
                    let shifted_r = e.select(&inside, &shifted_r, &initial[2]);
                    let shifted_i = e.select(&inside, &shifted_i, &zero_matrix);
                    let q = e.eig_qr(&(shifted_r, shifted_i), &state[4], n);
                    let qr = e.eig_transpose(&q.0);
                    let qi = e.eig_transpose(&q.1);
                    let qh = (qr, e.neg(&qi));
                    let left = e.eig_matmul(&qh, &t);
                    let t = e.eig_matmul(&left, &q);
                    let z = e.eig_matmul(&z, &q);
                    vec![
                        e.select(&qr_step, &t.0, &state[0]),
                        e.select(&qr_step, &t.1, &state[1]),
                        e.select(&qr_step, &z.0, &state[2]),
                        e.select(&qr_step, &z.1, &state[3]),
                        state[4].clone(),
                        state[5].clone(),
                    ]
                })
            })
        });
        let state = state
            .iter()
            .enumerate()
            .map(|(i, v)| {
                self.axes_view(
                    v,
                    if i < 4 {
                        axes.clone()
                    } else {
                        scalar_axes.clone()
                    },
                )
            })
            .collect::<Vec<_>>();
        let t = (state[0].clone(), state[1].clone());
        let z = (state[2].clone(), state[3].clone());
        let vectors = self.eig_vectors(&t, &z, n, epsilon);
        let vr = self.select(&diagonal, &t.0, &zero_matrix);
        let vi = self.select(&diagonal, &t.1, &zero_matrix);
        let values = (
            self.reduce_sum_last_axis(&vr),
            self.reduce_sum_last_axis(&vi),
        );
        let av = self.eig_matmul(&normalized_input, &vectors);
        let lr = self.matrix_expand(&values.0, &vectors.0, batch + 1);
        let li = self.matrix_expand(&values.1, &vectors.1, batch + 1);
        let rr = self.mul(&vectors.0, &lr);
        let ii = self.mul(&vectors.1, &li);
        let ri = self.mul(&vectors.0, &li);
        let ir = self.mul(&vectors.1, &lr);
        let product_r = self.sub(&rr, &ii);
        let product_i = self.add(&ri, &ir);
        let error_r = self.sub(&av.0, &product_r);
        let error_i = self.sub(&av.1, &product_i);
        let error = self.eig_hypot(&error_r, &error_i);
        let per_pair_error = self.reduce_axis(
            "stablehlo.maximum",
            reduce_max_identity(self.dtype),
            &error,
            batch,
        );
        let error = self.reduce_max(&per_pair_error);
        let tolerance = self.scalar(512.0 * n as f64 * epsilon);
        let certified = self.compare("LE", &error, &tolerance);
        let one_i = self.int_value_const(1);
        let completed = self.compare("LE", &state[4], &one_i);
        let completed = self.and(&completed, &finite);
        let completed = self.and(&completed, &certified);
        self.record_runtime_error(&completed, 10);
        // Use each eigenpair's residual, rather than the largest matrix entry,
        // to distinguish a small complex pair beside a large real eigenvalue.
        let lambda_size = self.eig_hypot(&values.0, &values.1);
        let eps = self.scalar(epsilon);
        let rounding = self.mul(&lambda_size, &eps);
        let rounding = self.add(&rounding, &per_pair_error);
        let safety = self.scalar(64.0);
        let rounding = self.mul(&rounding, &safety);
        let imaginary = self.abs(&values.1);
        let real_values = self.compare("LE", &imaginary, &rounding);
        let real_spectrum = self.reduce_boolean("stablehlo.and", "true", &real_values);
        let nan = self.scalar(f64::NAN);
        let vr = self.mul(&values.0, &scale);
        let vi = self.mul(&values.1, &scale);
        Ok(Eigen {
            values: (
                self.select(&completed, &vr, &nan),
                self.select(&completed, &vi, &nan),
            ),
            vectors: (
                self.select(&completed, &vectors.0, &nan),
                self.select(&completed, &vectors.1, &nan),
            ),
            real_spectrum,
        })
    }
}
