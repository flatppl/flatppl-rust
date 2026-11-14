//! §09 `ext-linear-algebra.matexp`: scaling, Padé approximation and squaring.
//! Uses portable matrix operations and pivoted elimination, preserving reverse AD
//! at repeated eigenvalues. No eigendecomposition or external library call.

use super::*;

impl Emitter<'_> {
    pub(super) fn lower_matrix_exp_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra" || member != "matexp" {
            return None;
        }
        Some(self.emit_matrix_exp(id, args, broadcast))
    }

    fn emit_matrix_exp(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        let [arg] = args else {
            return Err(EmitError::at(id, "matexp requires one square matrix"));
        };
        let input = self.lower_node(*arg)?;
        let mut values = [self.typed_axes(*arg, input)];
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let input = self.convert(&values[0], ElemKind::Real);
        let MlirTy::Ranked(dims) = self.cell_ty(&input) else {
            return Err(EmitError::at(id, "matexp requires a static square matrix"));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(id, "matexp requires a static square matrix"));
        };
        if n != m || *n == 0 || *n > i32::MAX as u64 {
            return Err(EmitError::at(
                id,
                "matexp requires a nonempty square matrix",
            ));
        }
        let output = self.matrix_exp(id, &input)?;
        Ok(match parent {
            Some(parent) => {
                let frame = self.broadcast_frame.clone();
                let output = self.finish_broadcast(&output, &frame, parent.len());
                self.broadcast_frame = parent;
                output
            }
            None => output,
        })
    }

    fn matrix_exp(&mut self, id: NodeId, input: &Value) -> Result<Value, EmitError> {
        // Higham (2005), doi:10.1137/04061101X. Single precision needs degree7;
        // degree13 is the double-precision scaling case. Normalize by b0 so
        // polynomial intermediates do not inherit the large integer coefficients.
        let (coefficients, theta): (&[f64], f64) = match self.dtype {
            Dtype::F32 => (
                &[
                    17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.,
                ],
                3.925_724_783_138_66,
            ),
            Dtype::F64 => (
                &[
                    64764752532480000.,
                    32382376266240000.,
                    7771770303897600.,
                    1187353796428800.,
                    129060195264000.,
                    10559470521600.,
                    670442572800.,
                    33522128640.,
                    1323241920.,
                    40840800.,
                    960960.,
                    16380.,
                    182.,
                    1.,
                ],
                5.371920351148152,
            ),
        };
        let batch = self.batch_rank(input);
        let rows = self.axis_indices(input, batch);
        let cols = self.axis_indices(input, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let magnitude = self.abs(input);
        let maximum = self.reduce_max(&magnitude);
        let infinity = self.inf_like(&maximum);
        let finite = self.compare("LT", &maximum, &infinity);
        let zeros = self.constant_like(0.0, input);
        let input = self.select(&finite, input, &zeros);
        // NaN/Inf cannot become an unbounded loop count. Keep invalid cells out
        // of the solve too, then propagate NaN in the final result.
        let maximum = self.select(&finite, &maximum, &one);
        let positive = self.compare("GT", &maximum, &zero);
        let safe_maximum = self.select(&positive, &maximum, &one);
        let scaled = self.div(&input, &safe_maximum);
        let magnitude = self.abs(&scaled);
        let columns = self.reduce_axis("stablehlo.add", "0.000000e+00", &magnitude, batch);
        let norm = self.reduce_max(&columns);
        let norm = self.max(&norm, &one);
        let log_maximum = self.log2(&safe_maximum);
        let log_norm = self.log2(&norm);
        let log_norm = self.add(&log_maximum, &log_norm);
        let threshold = self.scalar(theta.log2());
        let log_scale = self.sub(&log_norm, &threshold);
        let log_scale = self.max(&log_scale, &zero);
        let count = self.ceil(&log_scale);
        let count = self.convert(&count, ElemKind::Int);
        let exponent = self.convert(&count, ElemKind::Real);
        // Two normal power-of-two factors also scale finite values near the
        // dtype limit without first flushing a subnormal reciprocal to zero.
        let half = self.scalar(0.5);
        let first = self.mul(&exponent, &half);
        let first = self.floor(&first);
        let second = self.sub(&exponent, &first);
        let first = self.neg(&first);
        let second = self.neg(&second);
        let two = self.scalar(2.0);
        let first = self.pow(&two, &first);
        let second = self.pow(&two, &second);
        let a = self.mul(&input, &first);
        let a = self.mul(&a, &second);
        let a2 = self.matmat(&a, &a);
        let a4 = self.matmat(&a2, &a2);
        let a6 = self.matmat(&a4, &a2);
        let b = coefficients
            .iter()
            .map(|x| x / coefficients[0])
            .collect::<Vec<_>>();
        let combination = |e: &mut Self, terms: &[(&Value, f64)]| {
            let mut result = e.constant_like(0.0, &identity);
            for &(power, coefficient) in terms {
                let coefficient = e.scalar(coefficient);
                let term = e.mul(power, &coefficient);
                result = e.add(&result, &term);
            }
            result
        };
        let (u, v) = if b.len() == 8 {
            let odd = combination(
                self,
                &[(&a6, b[7]), (&a4, b[5]), (&a2, b[3]), (&identity, b[1])],
            );
            let even = combination(
                self,
                &[(&a6, b[6]), (&a4, b[4]), (&a2, b[2]), (&identity, b[0])],
            );
            (self.matmat(&a, &odd), even)
        } else {
            let odd = combination(self, &[(&a6, b[13]), (&a4, b[11]), (&a2, b[9])]);
            let odd = self.matmat(&a6, &odd);
            let tail = combination(
                self,
                &[(&a6, b[7]), (&a4, b[5]), (&a2, b[3]), (&identity, b[1])],
            );
            let odd = self.add(&odd, &tail);
            let even = combination(self, &[(&a6, b[12]), (&a4, b[10]), (&a2, b[8])]);
            let even = self.matmat(&a6, &even);
            let tail = combination(
                self,
                &[(&a6, b[6]), (&a4, b[4]), (&a2, b[2]), (&identity, b[0])],
            );
            (self.matmat(&a, &odd), self.add(&even, &tail))
        };
        let denominator = self.sub(&v, &u);
        let numerator = self.add(&v, &u);
        let initial = self.matrix_solve(id, &denominator, &numerator)?;
        // Fixed trip counts permit statically shaped reverse-AD tapes. Keep
        // the common 16-square range separate from the rare large-norm tail.
        // norm1(A) <= n*maxabs(A), so this bound covers every finite input.
        let n = shape(&input.ty)[batch].unwrap();
        let max_exponent = match self.dtype {
            Dtype::F32 => 128,
            Dtype::F64 => 1024,
        };
        let bound = max_exponent + n.next_power_of_two().ilog2() as u64;
        let result = self.matrix_exp_squares(&initial, &identity, &count, 0, 16);
        let sixteen = self.int_value_const(16);
        let tail = self.compare("GT", &count, &sixteen);
        let result = self.matrix_exp_if(&tail, &result, |e| {
            e.matrix_exp_squares(&result, &identity, &count, 16, bound)
        });
        let nan = self.constant_like(f64::NAN, &result);
        Ok(self.select(&finite, &result, &nan))
    }

    fn matrix_exp_squares(
        &mut self,
        initial: &Value,
        identity: &Value,
        count: &Value,
        start: u64,
        end: u64,
    ) -> Value {
        let axes = self.axes_of(initial);
        let result = self
            .matrix_loop(
                end - start,
                0,
                std::slice::from_ref(initial),
                |e, index, values| {
                    let offset = e.int_value_const(start as i64);
                    let index = e.add(index, &offset);
                    let active = e.compare("LT", &index, count);
                    let value = e.axes_view(&values[0], axes.clone());
                    let square = e.matrix_exp_if(&active, &value, |e| {
                        // A completed batch cell must not overflow in a discarded square.
                        let safe = e.select(&active, &value, identity);
                        let square = e.matmat(&safe, &safe);
                        e.select(&active, &square, &value)
                    });
                    vec![square]
                },
            )
            .remove(0);
        self.axes_view(&result, axes)
    }

    /// Lazy branch local to squaring. Neither branch contains runtime checks.
    fn matrix_exp_if(
        &mut self,
        active: &Value,
        initial: &Value,
        branch: impl FnOnce(&mut Self) -> Value,
    ) -> Value {
        let active = if matches!(active.ty, MlirTy::Scalar) {
            active.clone()
        } else {
            let flat = self.axes_view(
                active,
                Axes {
                    batch: 0,
                    layers: vec![shape(&active.ty).len()],
                },
            );
            self.reduce_boolean("stablehlo.or", "false", &flat)
        };
        let index = self.pure(format!(
            "stablehlo.convert {} : (tensor<i1>) -> tensor<i32>",
            active.ssa
        ));
        let saved = std::mem::take(&mut self.body);
        let saved_pure = std::mem::take(&mut self.pure_ops);
        let saved_expanded = std::mem::take(&mut self.expanded);
        self.region_depth += 1;
        let result = branch(self);
        self.region_depth -= 1;
        let body = std::mem::replace(&mut self.body, saved);
        self.pure_ops = saved_pure;
        self.expanded = saved_expanded;
        let ty = initial.ty.render(self.dtype, initial.elem);
        let ssa = self.fresh();
        self.push(&format!("{ssa} = \"stablehlo.case\"({index}) ({{\n stablehlo.return {} : {ty}\n}}, {{\n{body}\n stablehlo.return {} : {ty}\n}}) : (tensor<i32>) -> {ty}", initial.ssa, result.ssa));
        self.remember_axes(&ssa, self.axes_of(initial));
        Value {
            ssa,
            ..initial.clone()
        }
    }
}
