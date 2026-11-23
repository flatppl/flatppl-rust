//! Hermitian cyclic Jacobi iteration with ascending, orthonormal eigenpairs.

use super::*;

type Pair = (Value, Value);

pub(super) struct HermitianEigen {
    pub(super) values: Value,
    pub(super) vectors: Option<Pair>,
}

impl Emitter<'_> {
    fn hermitian_adjoint(&mut self, a: &Pair) -> Pair {
        let re = self.svd_transpose(&a.0);
        let im = self.svd_transpose(&a.1);
        (re, self.neg(&im))
    }

    fn hermitian_canonical_phase(&mut self, vectors: &Pair) -> Pair {
        let rt = self.svd_transpose(&vectors.0);
        let it = self.svd_transpose(&vectors.1);
        let rr = self.mul(&rt, &rt);
        let ii = self.mul(&it, &it);
        let magnitude = self.add(&rr, &ii);
        let largest = self.reduce_max_last_axis(&magnitude);
        let largest = self.svd_expand(&largest, &rt, false);
        let selected = self.compare("EQ", &magnitude, &largest);
        let indices = self.axis_indices(&rt, self.batch_rank(&rt) + 1);
        let limit = self.int_value_const(i32::MAX as i64);
        let candidates = self.select(&selected, &indices, &limit);
        let pivot = self.reduce_axis_lit(
            "stablehlo.minimum",
            &i32::MAX.to_string(),
            &candidates,
            self.batch_rank(&rt) + 1,
        );
        let pivot = self.svd_expand(&pivot, &rt, false);
        let selected = self.compare("EQ", &indices, &pivot);
        let zero = self.constant_like(0.0, &rt);
        let pr = self.select(&selected, &rt, &zero);
        let pi = self.select(&selected, &it, &zero);
        let pr = self.reduce_sum_last_axis(&pr);
        let pi = self.reduce_sum_last_axis(&pi);
        let rr = self.mul(&pr, &pr);
        let ii = self.mul(&pi, &pi);
        let norm = self.add(&rr, &ii);
        let norm = self.sqrt(&norm);
        let pr = self.div(&pr, &norm);
        let pi = self.div(&pi, &norm);
        let pr = self.svd_expand(&pr, &vectors.0, true);
        let pi = self.svd_expand(&pi, &vectors.0, true);
        let rr = self.mul(&vectors.0, &pr);
        let ii = self.mul(&vectors.1, &pi);
        let ri = self.mul(&vectors.1, &pr);
        let ir = self.mul(&vectors.0, &pi);
        (self.add(&rr, &ii), self.sub(&ri, &ir))
    }

    fn hermitian_entry(&mut self, a: &Value, p: &Value, q: &Value) -> Value {
        let column = self.svd_column(a, q);
        let rows = self.axis_indices(&column, self.batch_rank(&column));
        let selected = self.compare("EQ", &rows, p);
        let zero = self.constant_like(0.0, &column);
        let value = self.select(&selected, &column, &zero);
        self.reduce_sum_last_axis(&value)
    }

    /// The scaled Jacobi tangent avoids squaring a large diagonal difference.
    fn jacobi_rotation(
        &mut self,
        difference: &Value,
        coupling: &Value,
        active: &Value,
    ) -> (Value, Value, Value) {
        let one = self.scalar(1.0);
        let zero = self.scalar(0.0);
        let safe_gamma = self.select(active, coupling, &one);
        let two = self.scalar(2.0);
        let twice_gamma = self.mul(&two, &safe_gamma);
        let magnitude = self.abs(difference);
        let coupling_magnitude = self.abs(&twice_gamma);
        let scale = self.max(&magnitude, &coupling_magnitude);
        let d = self.div(&magnitude, &scale);
        let g = self.div(&twice_gamma, &scale);
        let d2 = self.mul(&d, &d);
        let g2 = self.mul(&g, &g);
        let radius = self.add(&d2, &g2);
        let radius = self.sqrt(&radius);
        let denominator = self.add(&d, &radius);
        let tangent = self.div(&g, &denominator);
        let negative = self.compare("LT", difference, &zero);
        let minus_tangent = self.neg(&tangent);
        let tangent = self.select(&negative, &minus_tangent, &tangent);
        let tangent = self.select(active, &tangent, &zero);
        let t2 = self.mul(&tangent, &tangent);
        let norm = self.add(&one, &t2);
        let norm = self.sqrt(&norm);
        let cosine = self.div(&one, &norm);
        (cosine, tangent, safe_gamma)
    }

    fn jacobi_panel_entry(&mut self, panel: &Value, row: &Value, col: &Value) -> Value {
        let ty = MlirTy::Ranked(vec![Some(1), Some(1)]);
        let panel_ty = panel.ty.render(self.dtype, panel.elem);
        let slice_ty = ty.render(self.dtype, panel.elem);
        let index_ty = row.ty.render(self.dtype, row.elem);
        let ssa = self.pure(format!(
            "stablehlo.dynamic_slice {}, {}, {}, sizes = [1, 1] : ({panel_ty}, {index_ty}, {index_ty}) -> {slice_ty}",
            panel.ssa, row.ssa, col.ssa,
        ));
        let entry = Value {
            ssa,
            ty,
            elem: panel.elem,
        };
        self.reshape(&entry, MlirTy::Scalar)
    }

    fn jacobi_update(&mut self, a: &Value, v: &Value, start: [&Value; 2]) -> Value {
        let [row, col] = start;
        let matrix_ty = a.ty.render(self.dtype, a.elem);
        let slice_ty = v.ty.render(self.dtype, v.elem);
        let index_ty = row.ty.render(self.dtype, row.elem);
        let ssa = self.pure(format!(
            "stablehlo.dynamic_update_slice {}, {}, {}, {} : ({matrix_ty}, {slice_ty}, {index_ty}, {index_ty}) -> {matrix_ty}",
            a.ssa, v.ssa, row.ssa, col.ssa,
        ));
        Value { ssa, ..a.clone() }
    }

    fn rotate_columns(
        &mut self,
        panel: &Value,
        n: u64,
        cosine: &Value,
        sine: &Value,
    ) -> (Value, Value) {
        let x = self.slice(panel, &[0, 0], &[n, 1], &[1, 1]);
        let y = self.slice(panel, &[0, 1], &[n, 2], &[1, 1]);
        let cx = self.mul(cosine, &x);
        let sy = self.mul(sine, &y);
        let sx = self.mul(sine, &x);
        let cy = self.mul(cosine, &y);
        (self.sub(&cx, &sy), self.add(&sx, &cy))
    }

    fn symmetric_jacobi(
        &mut self,
        n: u64,
        a: &Value,
        vectors: Option<&Value>,
        epsilon: f64,
    ) -> Vec<Value> {
        let mut matrices = vec![a.clone()];
        matrices.extend(vectors.cloned());
        if n > 1 {
            let zero = self.int_value_const(0);
            let one = self.int_value_const(1);
            let limit = self.int_value_const(n as i64);
            let eps = self.scalar(epsilon);
            let tolerance = self.scalar(64.0 * n as f64 * epsilon);
            let rows = self.axis_indices(a, 0);
            let cols = self.axis_indices(a, 1);
            let diagonal = self.compare("EQ", &rows, &cols);
            let zero_real = self.scalar(0.0);
            matrices.push(self.bool_value_const(true));
            matrices = self.matrix_loop(32, 0, &matrices, |e, _, state| {
                let (unfinished, matrices) = state.split_last().unwrap();
                e.matrix_if_values(unfinished, state, |e| {
                    // One final sweep after reaching the rotation tolerance retains
                    // the tangent of a separated eigenvector at exactly zero coupling.
                    let magnitude = e.abs(&matrices[0]);
                    let off = e.select(&diagonal, &zero_real, &magnitude);
                    let off = e.reduce_max(&off);
                    let unfinished = e.compare("GT", &off, &eps);
                    // Visit the strict upper triangle in row-major order.
                    let mut initial = vec![zero.clone(), one.clone()];
                    initial.extend_from_slice(matrices);
                    let result = e.matrix_loop(n * (n - 1) / 2, 0, &initial, |e, _, state| {
                        let (p, q, a) = (&state[0], &state[1], &state[2]);
                        // Keep only these columns live on the reverse-loop tape.
                        let indices = e.vector(&[p.clone(), q.clone()]);
                        let panel = e.gather_axis(a, &indices, 0, 1);
                        let alpha = e.jacobi_panel_entry(&panel, p, &zero);
                        let beta = e.jacobi_panel_entry(&panel, q, &one);
                        let gamma = e.jacobi_panel_entry(&panel, p, &one);
                        let absolute = e.abs(&gamma);
                        let difference = e.sub(&beta, &alpha);
                        let gap = e.abs(&difference);
                        let coupled = e.compare("GT", &absolute, &eps);
                        let separated = e.compare("GT", &gap, &tolerance);
                        let active = e.or(&coupled, &separated);
                        // The signed tangent is smooth at zero coupling for a separated
                        // diagonal pair. Its derivative rotates the corresponding eigenspaces.
                        let (c, t, _) = e.jacobi_rotation(&difference, &gamma, &active);
                        let s = e.mul(&c, &t);
                        let (cp, cq) = ([&zero, p], [&zero, q]);
                        let (x, y) = e.rotate_columns(&panel, n, &c, &s);
                        // Gᵀ A G has symmetric rows and columns. The pivot block
                        // becomes diag(α − tγ, β + tγ) on an active rotation.
                        let shift = e.mul(&t, &gamma);
                        let alpha = e.sub(&alpha, &shift);
                        let beta = e.add(&beta, &shift);
                        let origin = e.scalar(0.0);
                        let off = e.select(&active, &origin, &gamma);
                        let rows = e.axis_indices(&x, 0);
                        let at_p = e.compare("EQ", &rows, p);
                        let at_q = e.compare("EQ", &rows, q);
                        let x = e.select(&at_p, &alpha, &x);
                        let x = e.select(&at_q, &off, &x);
                        let y = e.select(&at_p, &off, &y);
                        let y = e.select(&at_q, &beta, &y);
                        let a = e.jacobi_update(a, &x, cp);
                        let a = e.jacobi_update(&a, &y, cq);
                        let xr = e.transpose(&x, &[1, 0]);
                        let yr = e.transpose(&y, &[1, 0]);
                        let a = e.jacobi_update(&a, &xr, [p, &zero]);
                        let both = e.jacobi_update(&a, &yr, [q, &zero]);
                        let next_q = e.add(q, &one);
                        let end_row = e.compare("EQ", &next_q, &limit);
                        let next_p = e.add(p, &one);
                        let first_q = e.add(&next_p, &one);
                        let mut next = vec![
                            e.select(&end_row, &next_p, p),
                            e.select(&end_row, &first_q, &next_q),
                            both,
                        ];
                        if let Some(v) = state.get(3) {
                            let panel = e.gather_axis(v, &indices, 0, 1);
                            let (x, y) = e.rotate_columns(&panel, n, &c, &s);
                            let v = e.jacobi_update(v, &x, cp);
                            next.push(e.jacobi_update(&v, &y, cq));
                        }
                        next
                    });
                    result.into_iter().skip(2).chain([unfinished]).collect()
                })
            });
            matrices.pop();
        }
        // Return the same component layout as the Hermitian iteration.
        matrices
            .into_iter()
            .flat_map(|v| {
                let zero = self.constant_like(0.0, &v);
                [v, zero]
            })
            .collect()
    }

    pub(super) fn matrix_hermitian(
        &mut self,
        id: NodeId,
        input: Pair,
        vectors: bool,
        real: bool,
    ) -> Result<HermitianEigen, EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(&input.0) else {
            return Err(EmitError::at(
                id,
                "Hermitian eigensolve requires a static square matrix",
            ));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(
                id,
                "Hermitian eigensolve requires static dimensions",
            ));
        };
        if n != m || *n > i32::MAX as u64 {
            return Err(EmitError::at(
                id,
                "Hermitian eigensolve requires a square matrix",
            ));
        }
        let n = *n;
        let input = (
            self.convert(&input.0, ElemKind::Real),
            self.convert(&input.1, ElemKind::Real),
        );
        let axes = self.axes_of(&input.0);
        let batch = self.batch_rank(&input.0);
        let rows = self.axis_indices(&input.0, batch);
        let cols = self.axis_indices(&input.0, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let zero_matrix = self.constant_like(0.0, &input.0);
        if n == 0 {
            let values = self.diagonal_axes(&input.0, batch, batch + 1);
            return Ok(HermitianEigen {
                values,
                vectors: vectors.then_some((identity, zero_matrix)),
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
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let safe_scale = self.select(&positive, &scale, &one);
        let scaled = (
            self.div(&input.0, &safe_scale),
            self.div(&input.1, &safe_scale),
        );
        let adjoint = self.hermitian_adjoint(&scaled);
        let dr = self.sub(&scaled.0, &adjoint.0);
        let di = self.sub(&scaled.1, &adjoint.1);
        let dr = self.abs(&dr);
        let di = self.abs(&di);
        let error = self.max(&dr, &di);
        let error = self.reduce_max(&error);
        let tolerance = self.scalar(64.0 * epsilon * n as f64);
        let hermitian = self.compare("LE", &error, &tolerance);
        self.record_runtime_error(&hermitian, 12);
        let half = self.scalar(0.5);
        let re = self.add(&scaled.0, &adjoint.0);
        let im = self.add(&scaled.1, &adjoint.1);
        let re = self.mul(&half, &re);
        let im = self.mul(&half, &im);
        let mut initial = vec![re, im];
        if vectors {
            initial.extend([identity.clone(), zero_matrix]);
        }
        let local = real && batch == 0;
        let result = if local {
            self.symmetric_jacobi(n, &initial[0], initial.get(2), epsilon)
        } else {
            self.matrix_loop(32, 0, &initial, |e, _, state| {
                e.matrix_loop(n * n, 0, state, |e, index, state| {
                    let state: Vec<_> =
                        state.iter().map(|v| e.axes_view(v, axes.clone())).collect();
                    let size = e.int_value_const(n as i64);
                    let p = e.trunc_div(index, &size);
                    let q = e.rem(index, &size);
                    let a = (state[0].clone(), state[1].clone());
                    let alpha = e.hermitian_entry(&a.0, &p, &p);
                    let beta = e.hermitian_entry(&a.0, &q, &q);
                    let gamma = (
                        e.hermitian_entry(&a.0, &p, &q),
                        e.hermitian_entry(&a.1, &p, &q),
                    );
                    let gr = e.abs(&gamma.0);
                    let gi = e.abs(&gamma.1);
                    let gamma_scale = e.max(&gr, &gi);
                    let nonzero = e.compare("GT", &gamma_scale, &zero);
                    let safe = e.select(&nonzero, &gamma_scale, &one);
                    let gr = e.div(&gamma.0, &safe);
                    let gi = e.div(&gamma.1, &safe);
                    let gr2 = e.mul(&gr, &gr);
                    let gi2 = e.mul(&gi, &gi);
                    let norm2 = e.add(&gr2, &gi2);
                    let absolute = e.svd_safe_sqrt(&norm2);
                    let absolute = e.mul(&absolute, &gamma_scale);
                    let eps = e.scalar(epsilon);
                    let correlated = e.compare("GT", &absolute, &eps);
                    let ordered = e.compare("LT", &p, &q);
                    let active = e.and(&ordered, &correlated);
                    let difference = e.sub(&beta, &alpha);
                    let (cosine, tangent, safe_gamma) =
                        e.jacobi_rotation(&difference, &absolute, &active);
                    let sine = e.mul(&cosine, &tangent);
                    let phase = (e.div(&gamma.0, &safe_gamma), e.div(&gamma.1, &safe_gamma));
                    let right = e.svd_rotate(&a, &p, &q, &cosine, &sine, &phase);
                    let adjoint = e.hermitian_adjoint(&right);
                    let both = e.svd_rotate(&adjoint, &p, &q, &cosine, &sine, &phase);
                    let both = e.hermitian_adjoint(&both);
                    let mut next = vec![both.0, both.1];
                    if vectors {
                        let v = e.svd_rotate(
                            &(state[2].clone(), state[3].clone()),
                            &p,
                            &q,
                            &cosine,
                            &sine,
                            &phase,
                        );
                        next.extend([v.0, v.1]);
                    }
                    next
                })
            })
        };
        let state: Vec<_> = result
            .iter()
            .map(|v| self.axes_view(v, axes.clone()))
            .collect();
        let values = self.diagonal_axes(&state[0], batch, batch + 1);
        let zero_matrix = self.constant_like(0.0, &state[0]);
        let off_r = self.select(&diagonal, &zero_matrix, &state[0]);
        let off_i = self.select(&diagonal, &zero_matrix, &state[1]);
        let off_r = self.abs(&off_r);
        let off_i = self.abs(&off_i);
        let off = self.max(&off_r, &off_i);
        let error = self.reduce_max(&off);
        let converged = self.compare("LE", &error, &tolerance);
        self.record_runtime_error(&converged, 10);
        let valid = self.and(&hermitian, &converged);
        // Stable sorting by pairwise rank retains one column per equal value.
        let column_values = self.svd_expand(&values, &identity, false);
        let row_values = self.svd_expand(&values, &identity, true);
        let less = self.compare("LT", &column_values, &row_values);
        let equal = self.compare("EQ", &column_values, &row_values);
        let before = self.compare("LT", &rows, &cols);
        let tied = self.and(&equal, &before);
        let earlier = self.or(&less, &tied);
        let earlier = self.convert(&earlier, ElemKind::Int);
        let ranks = self.reduce_axis("stablehlo.add", "0", &earlier, batch);
        let ranks = self.svd_expand(&ranks, &identity, false);
        let sorted = self.compare("EQ", &ranks, &cols);
        let permutation = self.convert(&sorted, ElemKind::Real);
        let permutation_t = self.svd_transpose(&permutation);
        let values = self.mul(&values, if local { &safe_scale } else { &scale });
        let values = self.matvec(&permutation_t, &values);
        let nan = self.scalar(f64::NAN);
        let values = self.select(&valid, &values, &nan);
        let vectors = if vectors {
            let re = self.matmat(&state[2], &permutation);
            let im = self.matmat(&state[3], &permutation);
            let (re, im) = self.hermitian_canonical_phase(&(re, im));
            Some((
                self.select(&valid, &re, &nan),
                self.select(&valid, &im, &nan),
            ))
        } else {
            None
        };
        Ok(HermitianEigen { values, vectors })
    }
}
