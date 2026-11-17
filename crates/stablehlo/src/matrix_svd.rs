//! One-sided cyclic Jacobi SVD with unitary column rotations. Scaling avoids
//! squaring the input's condition number, as an eigensolve of AᴴA would do.

use super::*;

type Pair = (Value, Value);

pub(super) struct Svd {
    pub(super) u: Pair,
    pub(super) s: Value,
    pub(super) v: Pair,
}

impl Emitter<'_> {
    fn svd_call(&self, id: NodeId) -> Option<(String, Vec<NodeId>, bool)> {
        let id = self.resolve_tuple_projections(id);
        let Node::Call(call) = self.node(id) else {
            return None;
        };
        let (callee, args, broadcast) = match call.head {
            CallHead::User(callee) => (callee, call.args.to_vec(), false),
            CallHead::Builtin(head) if self.resolve(head) == "broadcast" => {
                let (callee, args) = call.args.split_first()?;
                (*callee, args.to_vec(), true)
            }
            _ => return None,
        };
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        (module == "ext-linear-algebra"
            && matches!(member.as_str(), "svd" | "rank" | "lstsq" | "pinv" | "cond"))
        .then_some((member, args, broadcast))
    }

    fn emit_svd(&mut self, id: NodeId, arg: NodeId, broadcast: bool) -> Result<Svd, EmitError> {
        let (re, im) = crate::complex::parts(self, arg, &mut std::collections::HashMap::new())?;
        let (re, im) = self.broadcast_pair(&re, &im);
        let re = self.typed_axes(arg, re);
        let im = self.typed_axes(arg, im);
        let mut values = [re, im];
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let result = self.matrix_svd(id, (values[0].clone(), values[1].clone()))?;
        Ok(if let Some(parent) = parent {
            let frame = self.broadcast_frame.clone();
            let u = (
                self.finish_broadcast(&result.u.0, &frame, parent.len()),
                self.finish_broadcast(&result.u.1, &frame, parent.len()),
            );
            let s = self.finish_broadcast(&result.s, &frame, parent.len());
            let v = (
                self.finish_broadcast(&result.v.0, &frame, parent.len()),
                self.finish_broadcast(&result.v.1, &frame, parent.len()),
            );
            self.broadcast_frame = parent;
            Svd { u, s, v }
        } else {
            result
        })
    }

    pub(super) fn lower_svd_record(
        &mut self,
        id: NodeId,
        field: &str,
    ) -> Option<Result<Lowered, EmitError>> {
        let (member, args, broadcast) = self.svd_call(id)?;
        if member != "svd" {
            return None;
        }
        Some((|| {
            let [arg] = args.as_slice() else {
                return Err(EmitError::at(id, "svd requires one matrix"));
            };
            let svd = self.emit_svd(id, *arg, broadcast)?;
            let complex = crate::complex::complex_leaf(self.type_of(*arg));
            if !complex {
                self.bind_field(id, "U".to_owned(), Lowered::Tensor(svd.u.0.clone()));
                self.bind_field(id, "S".to_owned(), Lowered::Tensor(svd.s.clone()));
                self.bind_field(id, "V".to_owned(), Lowered::Tensor(svd.v.0.clone()));
            }
            if field == "S" {
                return Ok(Lowered::Tensor(svd.s));
            }
            if complex {
                return Err(EmitError::at(
                    id,
                    "complex singular vectors require a real or imaginary projection",
                ));
            }
            match field {
                "U" => Ok(Lowered::Tensor(svd.u.0)),
                "V" => Ok(Lowered::Tensor(svd.v.0)),
                _ => Err(EmitError::at(id, "svd has fields U, S and V")),
            }
        })())
    }

    pub(crate) fn lower_svd_field_parts(
        &mut self,
        id: NodeId,
        container: NodeId,
        field: &str,
    ) -> Option<Result<Pair, EmitError>> {
        let (member, args, broadcast) = self.svd_call(container)?;
        if member != "svd" {
            return None;
        }
        Some((|| {
            let [arg] = args.as_slice() else {
                return Err(EmitError::at(id, "svd requires one matrix"));
            };
            let svd = self.emit_svd(container, *arg, broadcast)?;
            match field {
                "U" => Ok(svd.u),
                "V" => Ok(svd.v),
                "S" => Ok((svd.s, self.scalar(0.0))),
                _ => Err(EmitError::at(id, "svd has fields U, S and V")),
            }
        })())
    }

    pub(super) fn lower_svd_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra"
            || !matches!(member.as_str(), "rank" | "lstsq" | "pinv" | "cond")
        {
            return None;
        }
        Some((|| {
            if matches!(member.as_str(), "pinv" | "lstsq")
                && crate::complex::complex_leaf(self.type_of(id))
            {
                return Err(EmitError::at(
                    id,
                    "complex matrix result requires real/imaginary projections",
                ));
            }
            Ok(self.emit_svd_operation(id, &member, args, broadcast)?.0)
        })())
    }

    pub(crate) fn lower_lstsq_parts(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Pair, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra" || !matches!(member.as_str(), "lstsq" | "pinv") {
            return None;
        }
        Some(self.emit_svd_operation(id, &member, args, broadcast))
    }

    fn emit_svd_operation(
        &mut self,
        id: NodeId,
        member: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Pair, EmitError> {
        let Some(&a) = args.first() else {
            return Err(EmitError::at(id, "matrix operation requires a matrix"));
        };
        let mut memo = std::collections::HashMap::new();
        let (ar, ai) = crate::complex::parts(self, a, &mut memo)?;
        let (ar, ai) = self.broadcast_pair(&ar, &ai);
        let mut values = vec![self.typed_axes(a, ar), self.typed_axes(a, ai)];
        let option_start = if member == "lstsq" {
            let Some(&b) = args.get(1) else {
                return Err(EmitError::at(id, "lstsq requires a RHS"));
            };
            let (br, bi) = crate::complex::parts(self, b, &mut memo)?;
            let (br, bi) = self.broadcast_pair(&br, &bi);
            values.extend([self.typed_axes(b, br), self.typed_axes(b, bi)]);
            2
        } else {
            1
        };
        let cutoff_index = values.len();
        if member != "cond" {
            let rtol = if let Some(&arg) = args.get(option_start) {
                self.lower_node(arg)?
            } else {
                self.scalar(-1.0)
            };
            let atol = if let Some(&arg) = args.get(option_start + 1) {
                self.lower_node(arg)?
            } else {
                self.scalar(0.0)
            };
            values.extend([rtol, atol]);
        }
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let MlirTy::Ranked(dims) = self.cell_ty(&values[0]) else {
            return Err(EmitError::at(id, "operation requires a static matrix"));
        };
        let [Some(m), Some(n)] = dims.as_slice() else {
            return Err(EmitError::at(
                id,
                "operation requires static matrix dimensions",
            ));
        };
        let (m, n) = (*m, *n);
        if member == "cond" && m.min(n) == 0 {
            return Err(EmitError::at(
                id,
                "cond requires a nonempty singular spectrum",
            ));
        }
        let input_finite = if matches!(member, "pinv" | "cond") {
            let finite = self.finite_matrix_parts(&(values[0].clone(), values[1].clone()));
            self.record_runtime_error(&finite, 12);
            finite
        } else {
            // Existing rank/lstsq retain SVD's numerical-failure status.
            self.bool_value_const(true)
        };
        let zeros = self.constant_like(0.0, &values[0]);
        let safe_input = (
            self.select(&input_finite, &values[0], &zeros),
            self.select(&input_finite, &values[1], &zeros),
        );
        let svd = self.matrix_svd_impl(id, safe_input, matches!(member, "pinv" | "lstsq"))?;
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let (result, valid) = if member == "cond" {
            let maximum = self.reduce_max(&svd.s);
            let minimum = self.reduce_min(&svd.s);
            let nonzero = self.compare("GT", &minimum, &zero);
            let safe = self.select(&nonzero, &minimum, &one);
            let ratio = self.div(&maximum, &safe);
            let infinity = self.scalar(f64::INFINITY);
            let condition = self.select(&nonzero, &ratio, &infinity);
            ((condition, zero.clone()), self.bool_value_const(true))
        } else {
            let epsilon = match self.dtype {
                Dtype::F32 => f32::EPSILON as f64,
                Dtype::F64 => f64::EPSILON,
            };
            let auto = self.scalar(epsilon * m.max(n) as f64);
            let sentinel = self.scalar(-1.0);
            let rtol = &values[cutoff_index];
            let atol = &values[cutoff_index + 1];
            let automatic = self.compare("EQ", rtol, &sentinel);
            let nonnegative = self.compare("GE", rtol, &zero);
            let infinity = self.scalar(f64::INFINITY);
            let finite = self.compare("LT", rtol, &infinity);
            let ordinary = self.and(&nonnegative, &finite);
            let valid_rtol = self.or(&automatic, &ordinary);
            let nonnegative = self.compare("GE", atol, &zero);
            let finite = self.compare("LT", atol, &infinity);
            let valid_atol = self.and(&nonnegative, &finite);
            let valid = self.and(&valid_rtol, &valid_atol);
            self.record_runtime_error(&valid, 12);
            let rtol = self.select(&automatic, &auto, rtol);
            let maximum = self.reduce_max(&svd.s);
            let relative = self.mul(&rtol, &maximum);
            let cutoff = self.max(atol, &relative);
            let retained = self.compare("GT", &svd.s, &cutoff);
            if member == "rank" {
                let retained = self.convert(&retained, ElemKind::Int);
                let rank = self.reduce_sum(&retained);
                ((rank, zero.clone()), valid)
            } else {
                let safe_s = self.select(&retained, &svd.s, &one);
                let reciprocal = self.div(&one, &safe_s);
                let reciprocal = self.select(&retained, &reciprocal, &zero);
                let u_adjoint = self.factor_adjoint(&svd.u);
                let result = if member == "lstsq" {
                    let MlirTy::Ranked(bdims) = self.cell_ty(&values[2]) else {
                        return Err(EmitError::at(id, "lstsq requires a vector or matrix RHS"));
                    };
                    if bdims.first() != Some(&Some(m)) || !(1..=2).contains(&bdims.len()) {
                        return Err(EmitError::at(id, "lstsq RHS rows must equal matrix rows"));
                    }
                    let matrix_rhs = bdims.len() == 2;
                    let b = (values[2].clone(), values[3].clone());
                    // Apply Uᴴ to the RHS first; forming the full pseudoinverse
                    // would add an n×m workspace even for one RHS vector.
                    let projected = self.factor_apply(&u_adjoint, &b);
                    let scale = if matrix_rhs {
                        self.svd_expand(&reciprocal, &projected.0, false)
                    } else {
                        reciprocal
                    };
                    let projected = (
                        self.mul(&projected.0, &scale),
                        self.mul(&projected.1, &scale),
                    );
                    self.factor_apply(&svd.v, &projected)
                } else {
                    let scale = self.svd_expand(&reciprocal, &svd.v.0, true);
                    let v = (self.mul(&svd.v.0, &scale), self.mul(&svd.v.1, &scale));
                    self.factor_matmul(&v, &u_adjoint)
                };
                (result, valid)
            }
        };
        let valid = self.and(&valid, &input_finite);
        let result = if member == "rank" {
            result
        } else {
            let nan = self.scalar(f64::NAN);
            (
                self.select(&valid, &result.0, &nan),
                self.select(&valid, &result.1, &nan),
            )
        };
        Ok(if let Some(parent) = parent {
            let frame = self.broadcast_frame.clone();
            let re = self.finish_broadcast(&result.0, &frame, parent.len());
            let im = self.finish_broadcast(&result.1, &frame, parent.len());
            self.broadcast_frame = parent;
            (re, im)
        } else {
            result
        })
    }

    pub(super) fn svd_transpose(&mut self, a: &Value) -> Value {
        let batch = self.batch_rank(a);
        let mut permutation: Vec<_> = (0..batch as u64).collect();
        permutation.extend([batch as u64 + 1, batch as u64]);
        let value = self.transpose(a, &permutation);
        self.axes_view(&value, self.axes_of(a))
    }

    pub(super) fn svd_expand(&mut self, vector: &Value, matrix: &Value, row: bool) -> Value {
        let batch = self.batch_rank(matrix);
        let axis = batch + usize::from(row);
        let map: Vec<_> = (0..batch as u64).chain([axis as u64]).collect();
        self.expand_axes(vector, &map, matrix.ty.clone(), self.axes_of(matrix))
    }

    pub(super) fn svd_column(&mut self, matrix: &Value, index: &Value) -> Value {
        let columns = self.axis_indices(matrix, self.batch_rank(matrix) + 1);
        let mask = self.compare("EQ", &columns, index);
        let zero = self.constant_like(0.0, matrix);
        let column = self.select(&mask, matrix, &zero);
        self.reduce_sum_last_axis(&column)
    }

    pub(super) fn svd_safe_sqrt(&mut self, x: &Value) -> Value {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", x, &zero);
        let safe = self.select(&positive, x, &one);
        let root = self.sqrt(&safe);
        self.select(&positive, &root, &zero)
    }

    fn svd_norm2(&mut self, a: &Pair) -> Value {
        let re = self.mul(&a.0, &a.0);
        let im = self.mul(&a.1, &a.1);
        let square = self.add(&re, &im);
        self.reduce_sum(&square)
    }

    fn svd_inner(&mut self, a: &Pair, b: &Pair) -> Pair {
        let rr = self.mul(&a.0, &b.0);
        let ii = self.mul(&a.1, &b.1);
        let ri = self.mul(&a.0, &b.1);
        let ir = self.mul(&a.1, &b.0);
        let re = self.add(&rr, &ii);
        let im = self.sub(&ri, &ir);
        (self.reduce_sum(&re), self.reduce_sum(&im))
    }

    /// Apply a unitary two-column rotation to B and its right singular basis.
    pub(super) fn svd_rotate(
        &mut self,
        matrix: &Pair,
        p: &Value,
        q: &Value,
        cosine: &Value,
        sine: &Value,
        phase: &Pair,
    ) -> Pair {
        let ap = (self.svd_column(&matrix.0, p), self.svd_column(&matrix.1, p));
        let aq = (self.svd_column(&matrix.0, q), self.svd_column(&matrix.1, q));
        // p' = c p - s conj(phase) q; q' = s phase p + c q.
        let rr = self.mul(&phase.0, &aq.0);
        let ii = self.mul(&phase.1, &aq.1);
        let qr = self.add(&rr, &ii);
        let ri = self.mul(&phase.0, &aq.1);
        let ir = self.mul(&phase.1, &aq.0);
        let qi = self.sub(&ri, &ir);
        let rr = self.mul(&phase.0, &ap.0);
        let ii = self.mul(&phase.1, &ap.1);
        let pr = self.sub(&rr, &ii);
        let ri = self.mul(&phase.0, &ap.1);
        let ir = self.mul(&phase.1, &ap.0);
        let pi = self.add(&ri, &ir);
        let cp = self.mul(cosine, &ap.0);
        let sq = self.mul(sine, &qr);
        let new_pr = self.sub(&cp, &sq);
        let cp = self.mul(cosine, &ap.1);
        let sq = self.mul(sine, &qi);
        let new_pi = self.sub(&cp, &sq);
        let sp = self.mul(sine, &pr);
        let cq = self.mul(cosine, &aq.0);
        let new_qr = self.add(&sp, &cq);
        let sp = self.mul(sine, &pi);
        let cq = self.mul(cosine, &aq.1);
        let new_qi = self.add(&sp, &cq);
        let columns = self.axis_indices(&matrix.0, self.batch_rank(&matrix.0) + 1);
        let at_p = self.compare("EQ", &columns, p);
        let at_q = self.compare("EQ", &columns, q);
        let new_pr = self.svd_expand(&new_pr, &matrix.0, false);
        let new_pi = self.svd_expand(&new_pi, &matrix.1, false);
        let new_qr = self.svd_expand(&new_qr, &matrix.0, false);
        let new_qi = self.svd_expand(&new_qi, &matrix.1, false);
        let re = self.select(&at_p, &new_pr, &matrix.0);
        let re = self.select(&at_q, &new_qr, &re);
        let im = self.select(&at_p, &new_pi, &matrix.1);
        let im = self.select(&at_q, &new_qi, &im);
        (re, im)
    }

    pub(super) fn matrix_svd(&mut self, id: NodeId, input: Pair) -> Result<Svd, EmitError> {
        self.matrix_svd_impl(id, input, false)
    }

    fn matrix_svd_impl(
        &mut self,
        id: NodeId,
        input: Pair,
        preserve_small_modes: bool,
    ) -> Result<Svd, EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(&input.0) else {
            return Err(EmitError::at(id, "svd requires a static matrix"));
        };
        let [Some(m), Some(n)] = dims.as_slice() else {
            return Err(EmitError::at(id, "svd requires static matrix dimensions"));
        };
        if *m == 0 || *n == 0 {
            let batch = self.batch_rank(&input.0);
            let prefix = shape(&input.0.ty)[..batch].to_vec();
            let empty = |e: &mut Self, cell: &[u64]| {
                let mut dims = prefix.clone();
                dims.extend(cell.iter().copied().map(Some));
                let value = e.constant(0.0, MlirTy::Ranked(dims));
                e.axes_view(
                    &value,
                    Axes {
                        batch,
                        layers: vec![cell.len()],
                    },
                )
            };
            let u = empty(self, &[*m, 0]);
            let v = empty(self, &[*n, 0]);
            let s = empty(self, &[0]);
            return Ok(Svd {
                u: (u.clone(), u),
                s,
                v: (v.clone(), v),
            });
        }
        if (*m).max(*n) > i32::MAX as u64 {
            return Err(EmitError::at(
                id,
                "svd matrix dimensions exceed the supported index width",
            ));
        }
        let wide = m < n;
        let input = if wide {
            let re = self.svd_transpose(&input.0);
            let im = self.svd_transpose(&input.1);
            (re, self.neg(&im))
        } else {
            input
        };
        let (m, n) = ((*m).max(*n), (*m).min(*n));
        let input = (
            self.convert(&input.0, ElemKind::Real),
            self.convert(&input.1, ElemKind::Real),
        );
        let batch = self.batch_rank(&input.0);
        let axes = self.axes_of(&input.0);
        let magnitude_r = self.abs(&input.0);
        let magnitude_i = self.abs(&input.1);
        let magnitude = self.max(&magnitude_r, &magnitude_i);
        let scale = self.reduce_max(&magnitude);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let safe_scale = self.select(&positive, &scale, &one);
        let b = (
            self.div(&input.0, &safe_scale),
            self.div(&input.1, &safe_scale),
        );
        let mut v_dims = shape(&input.0.ty)[..batch].to_vec();
        v_dims.extend([Some(n), Some(n)]);
        let v = self.constant(0.0, MlirTy::Ranked(v_dims));
        let v = self.axes_view(&v, axes.clone());
        let rows = self.axis_indices(&v, batch);
        let columns = self.axis_indices(&v, batch + 1);
        let diagonal = self.compare("EQ", &rows, &columns);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let epsilon = match self.dtype {
            Dtype::F32 => f32::EPSILON as f64,
            Dtype::F64 => f64::EPSILON,
        };
        let tolerance = self.scalar(4.0 * epsilon);
        let initial = [b.0, b.1, identity, v];
        // Static sweep bounds keep reverse AD's storage shapes known. A final
        // orthogonality certificate rejects matrices needing more sweeps.
        let result = self.matrix_loop(40, 0, &initial, |e, _, state| {
            e.matrix_loop(n * n, 0, state, |e, index, state| {
                let state: Vec<_> = state.iter().map(|v| e.axes_view(v, axes.clone())).collect();
                let size = e.int_value_const(n as i64);
                let p = e.trunc_div(index, &size);
                let q = e.rem(index, &size);
                let bp = (e.svd_column(&state[0], &p), e.svd_column(&state[1], &p));
                let bq = (e.svd_column(&state[0], &q), e.svd_column(&state[1], &q));
                // Global scaling does not protect a small correlated block:
                // its inner products can underflow while the modes remain
                // representable. A common pair scale preserves the rotation.
                let pr = e.abs(&bp.0);
                let pi = e.abs(&bp.1);
                let qr = e.abs(&bq.0);
                let qi = e.abs(&bq.1);
                let pm = e.max(&pr, &pi);
                let qm = e.max(&qr, &qi);
                let magnitude = e.max(&pm, &qm);
                let pair_scale = e.reduce_max(&magnitude);
                let positive = e.compare("GT", &pair_scale, &zero);
                let pair_scale = e.select(&positive, &pair_scale, &one);
                let bp = (e.div(&bp.0, &pair_scale), e.div(&bp.1, &pair_scale));
                let bq = (e.div(&bq.0, &pair_scale), e.div(&bq.1, &pair_scale));
                let alpha = e.svd_norm2(&bp);
                let beta = e.svd_norm2(&bq);
                let gamma = e.svd_inner(&bp, &bq);
                let gr = e.abs(&gamma.0);
                let gi = e.abs(&gamma.1);
                let gamma_scale = e.max(&gr, &gi);
                let nonzero_gamma = e.compare("GT", &gamma_scale, &zero);
                let safe = e.select(&nonzero_gamma, &gamma_scale, &one);
                let gr = e.div(&gr, &safe);
                let gi = e.div(&gi, &safe);
                let gr2 = e.mul(&gr, &gr);
                let gi2 = e.mul(&gi, &gi);
                let absolute2 = e.add(&gr2, &gi2);
                let absolute = e.svd_safe_sqrt(&absolute2);
                let absolute = e.mul(&absolute, &gamma_scale);
                let norm_p = e.svd_safe_sqrt(&alpha);
                let norm_q = e.svd_safe_sqrt(&beta);
                let norm_product = e.mul(&norm_p, &norm_q);
                let threshold = e.mul(&tolerance, &norm_product);
                let correlated = e.compare("GT", &absolute, &threshold);
                let ordered = e.compare("LT", &p, &q);
                let active = e.and(&ordered, &correlated);
                let safe_gamma = e.select(&active, &absolute, &one);
                let difference = e.sub(&beta, &alpha);
                let two = e.scalar(2.0);
                let twice_gamma = e.mul(&two, &safe_gamma);
                let magnitude = e.abs(&difference);
                let rotation_scale = e.max(&magnitude, &twice_gamma);
                let d = e.div(&magnitude, &rotation_scale);
                let g = e.div(&twice_gamma, &rotation_scale);
                let d2 = e.mul(&d, &d);
                let g2 = e.mul(&g, &g);
                let radius = e.add(&d2, &g2);
                let radius = e.sqrt(&radius);
                let denominator = e.add(&d, &radius);
                let tangent = e.div(&g, &denominator);
                let negative = e.compare("LT", &difference, &zero);
                let minus_tangent = e.neg(&tangent);
                let tangent = e.select(&negative, &minus_tangent, &tangent);
                let tangent = e.select(&active, &tangent, &zero);
                let t2 = e.mul(&tangent, &tangent);
                let norm = e.add(&one, &t2);
                let norm = e.sqrt(&norm);
                let cosine = e.div(&one, &norm);
                let sine = e.mul(&cosine, &tangent);
                let phase = (e.div(&gamma.0, &safe_gamma), e.div(&gamma.1, &safe_gamma));
                let b = e.svd_rotate(
                    &(state[0].clone(), state[1].clone()),
                    &p,
                    &q,
                    &cosine,
                    &sine,
                    &phase,
                );
                let v = e.svd_rotate(
                    &(state[2].clone(), state[3].clone()),
                    &p,
                    &q,
                    &cosine,
                    &sine,
                    &phase,
                );
                vec![b.0, b.1, v.0, v.1]
            })
        });
        let state: Vec<_> = result
            .iter()
            .map(|v| self.axes_view(v, axes.clone()))
            .collect();
        let bt = (self.svd_transpose(&state[0]), self.svd_transpose(&state[1]));
        let ar = self.abs(&bt.0);
        let ai = self.abs(&bt.1);
        let magnitude = self.max(&ar, &ai);
        let column_scale = self.reduce_max_last_axis(&magnitude);
        let positive = self.compare("GT", &column_scale, &zero);
        let safe = self.select(&positive, &column_scale, &one);
        let expanded = self.svd_expand(&safe, &bt.0, false);
        let br = self.div(&bt.0, &expanded);
        let bi = self.div(&bt.1, &expanded);
        let rr = self.mul(&br, &br);
        let ii = self.mul(&bi, &bi);
        let norm2 = self.add(&rr, &ii);
        let norm2 = self.reduce_sum_last_axis(&norm2);
        let norm = self.svd_safe_sqrt(&norm2);
        let norm = self.mul(&norm, &column_scale);
        let singular_values = self.mul(&norm, &scale);
        let largest = self.reduce_max(&norm);
        let cutoff = self.scalar(epsilon * m as f64);
        let cutoff = self.mul(&largest, &cutoff);
        let nonzero = self.compare("GT", &norm, &cutoff);
        let safe_norm = self.select(&nonzero, &norm, &one);
        let safe_norm = self.svd_expand(&safe_norm, &state[0], true);
        let u = (
            self.div(&state[0], &safe_norm),
            self.div(&state[1], &safe_norm),
        );
        let urt = self.svd_transpose(&u.0);
        let uit = self.svd_transpose(&u.1);
        let rr = self.matmat(&urt, &u.0);
        let ii = self.matmat(&uit, &u.1);
        let gram_re = self.add(&rr, &ii);
        let ri = self.matmat(&urt, &u.1);
        let ir = self.matmat(&uit, &u.0);
        let gram_im = self.sub(&ri, &ir);
        let zero_gram = self.constant_like(0.0, &gram_re);
        let off_re = self.select(&diagonal, &zero_gram, &gram_re);
        let off_im = self.select(&diagonal, &zero_gram, &gram_im);
        let abs_re = self.abs(&off_re);
        let abs_im = self.abs(&off_im);
        let error = self.max(&abs_re, &abs_im);
        let error = self.reduce_max(&error);
        let tolerance = self.scalar(64.0 * epsilon * n as f64);
        let converged = self.compare("LE", &error, &tolerance);
        self.record_runtime_error(&converged, 10);
        let u = if preserve_small_modes {
            // A caller may explicitly retain a nonzero mode below the default
            // rank threshold. Null-basis completion would erase its phase.
            let positive = self.compare("GT", &norm, &zero);
            let safe_norm = self.select(&positive, &norm, &one);
            let safe_norm = self.svd_expand(&safe_norm, &state[0], true);
            (
                self.div(&state[0], &safe_norm),
                self.div(&state[1], &safe_norm),
            )
        } else {
            self.svd_complete_columns(&u, &nonzero, n)
        };
        let nan = self.scalar(f64::NAN);
        let u = (
            self.select(&converged, &u.0, &nan),
            self.select(&converged, &u.1, &nan),
        );
        let singular_values = self.select(&converged, &singular_values, &nan);
        let v = (
            self.select(&converged, &state[2], &nan),
            self.select(&converged, &state[3], &nan),
        );
        Ok(if wide {
            Svd {
                u: v,
                s: singular_values,
                v: u,
            }
        } else {
            Svd {
                u,
                s: singular_values,
                v,
            }
        })
    }

    /// Complete numerically null columns to an orthonormal thin basis. The
    /// row with greatest unused squared length supplies a stable coordinate.
    fn svd_complete_columns(&mut self, input: &Pair, nonzero: &Value, n: u64) -> Pair {
        let axes = self.axes_of(&input.0);
        let batch = self.batch_rank(&input.0);
        let zero = self.constant_like(0.0, &input.0);
        let retained = self.svd_expand(nonzero, &input.0, true);
        let initial_re = self.select(&retained, &input.0, &zero);
        let initial_im = self.select(&retained, &input.1, &zero);
        let result = self.matrix_loop(n, 0, &[initial_re, initial_im], |e, index, state| {
            let re = e.axes_view(&state[0], axes.clone());
            let im = e.axes_view(&state[1], axes.clone());
            let rr = e.mul(&re, &re);
            let ii = e.mul(&im, &im);
            let used = e.add(&rr, &ii);
            let used = e.reduce_sum_last_axis(&used);
            let minimum = e.reduce_min(&used);
            let least_used = e.compare("EQ", &used, &minimum);
            let rows = e.axis_indices(&used, batch);
            let real_rows = e.convert(&rows, ElemKind::Real);
            let row_limit = e.scalar(i32::MAX as f64);
            let candidates = e.select(&least_used, &real_rows, &row_limit);
            let row = e.reduce_min(&candidates);
            let basis = e.compare("EQ", &real_rows, &row);
            let basis = e.convert(&basis, ElemKind::Real);
            let columns = e.axis_indices(nonzero, batch);
            let at_column = e.compare("EQ", &columns, index);
            let selected = e.and(nonzero, &at_column);
            let selected = e.convert(&selected, ElemKind::Real);
            let has_value = e.reduce_sum(&selected);
            let zero_scalar = e.scalar(0.0);
            let has_value = e.compare("GT", &has_value, &zero_scalar);
            let mut ar = basis;
            let mut ai = e.constant_like(0.0, &ar);
            // Two modified Gram-Schmidt passes protect null completions when
            // previous columns are nearly aligned with coordinate vectors.
            for _ in 0..2 {
                let rt = e.svd_transpose(&re);
                let it = e.svd_transpose(&im);
                let rr = e.matvec(&rt, &ar);
                let ii = e.matvec(&it, &ai);
                let pr = e.add(&rr, &ii);
                let ri = e.matvec(&rt, &ai);
                let ir = e.matvec(&it, &ar);
                let pi = e.sub(&ri, &ir);
                let rr = e.matvec(&re, &pr);
                let ii = e.matvec(&im, &pi);
                let dr = e.sub(&rr, &ii);
                let ri = e.matvec(&re, &pi);
                let ir = e.matvec(&im, &pr);
                let di = e.add(&ri, &ir);
                ar = e.sub(&ar, &dr);
                ai = e.sub(&ai, &di);
            }
            let norm2 = e.svd_norm2(&(ar.clone(), ai.clone()));
            let one = e.scalar(1.0);
            let safe_norm2 = e.select(&has_value, &one, &norm2);
            let norm = e.sqrt(&safe_norm2);
            let ar = e.div(&ar, &norm);
            let ai = e.div(&ai, &norm);
            let ar = e.svd_expand(&ar, &re, false);
            let ai = e.svd_expand(&ai, &im, false);
            let columns = e.axis_indices(&re, batch + 1);
            let selected = e.compare("EQ", &columns, index);
            let missing = e.not(&has_value);
            let selected = e.and(&selected, &missing);
            vec![e.select(&selected, &ar, &re), e.select(&selected, &ai, &im)]
        });
        (
            self.axes_view(&result[0], axes.clone()),
            self.axes_view(&result[1], axes),
        )
    }
}
