//! §09 `ext-linear-algebra.matexp`: scaling, polynomial approximation and squaring.
//! Uses portable matrix operations, preserving reverse AD
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
        let output = self.matrix_exp(&input);
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

    fn matrix_exp(&mut self, input: &Value) -> Value {
        // Conservative norm bounds for solve-free optimized Taylor polynomials.
        let theta: f64 = match self.dtype {
            Dtype::F32 => 3.94,
            Dtype::F64 => 1.09,
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
        // of the polynomial too, then propagate NaN in the final result.
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
        let initial = self.matrix_exp_polynomial(&a, &identity);
        // Fixed trip counts permit statically shaped reverse-AD tapes. Keep
        // the common 16-square range separate from the rare large-norm tail.
        // norm1(A) <= n*maxabs(A), so this bound covers every finite input.
        let n = shape(&input.ty)[batch].unwrap();
        let max_exponent = match self.dtype {
            Dtype::F32 => 128,
            Dtype::F64 => 1024,
        };
        let bound = max_exponent + n.next_power_of_two().ilog2() as u64;
        let zero_count = self.int_value_const(0);
        let needs_squaring = self.compare("GT", &count, &zero_count);
        let result = self.matrix_exp_if(&needs_squaring, &initial, |e| {
            if e.target == crate::Target::Cpu {
                return e.matrix_exp_squares(&initial, &identity, &count, 0, 16);
            }
            // GPU launch costs favor short guarded stages. On CPU their
            // reverse-AD branch tapes cost more than the inactive loop slots.
            let mut result = e.matrix_exp_squares(&initial, &identity, &count, 0, 1);
            for (start, end) in [(1, 2), (2, 4), (4, 8), (8, 16)] {
                let threshold = e.int_value_const(start as i64);
                let active = e.compare("GT", &count, &threshold);
                result = e.matrix_exp_if(&active, &result, |e| {
                    e.matrix_exp_squares(&result, &identity, &count, start, end)
                });
            }
            result
        });
        let sixteen = self.int_value_const(16);
        let tail = self.compare("GT", &count, &sixteen);
        let result = self.matrix_exp_if(&tail, &result, |e| {
            e.matrix_exp_squares(&result, &identity, &count, 16, bound)
        });
        let nan = self.constant_like(f64::NAN, &result);
        self.select(&finite, &result, &nan)
    }

    /// Solve-free t21^[24] and T18 factorizations, with five matrix products.
    /// Blanes et al. (2025), doi:10.1137/24M1657250, and Bader et al. (2019),
    /// doi:10.3390/math7121174. All factors are polynomials in the same matrix.
    fn matrix_exp_polynomial(&mut self, a: &Value, identity: &Value) -> Value {
        let a2 = self.matmat(a, a);
        let a3 = self.matmat(&a2, a);
        let combination = |e: &mut Self, terms: &[(&Value, f64)]| {
            let mut result = e.constant_like(0.0, identity);
            for &(power, coefficient) in terms {
                let coefficient = e.scalar(coefficient);
                let term = e.mul(power, &coefficient);
                result = e.add(&result, &term);
            }
            result
        };
        match self.dtype {
            Dtype::F32 => {
                let c = [
                    1.161_658_834_444_88e-6,
                    4.500_852_739_573_01e-6,
                    5.374708803114821e-5,
                    2.005403977292901e-3,
                    6.974348269544424e-2,
                    9.418613214806352e-1,
                    2.852960512714315e-3,
                    -7.544837153586671e-3,
                    1.829773504500424,
                    3.151382711608315e-2,
                    1.392249143769798e-1,
                    -2.269101241269351e-3,
                    -5.394098846866402e-2,
                    3.112216227982407e-1,
                    9.343851261938047,
                    6.865706355662834e-1,
                    3.233_370_163_085_38,
                    -5.726379787260966,
                    -1.413550099309667e-2,
                    -1.638413114712016e-1,
                ];
                let inner = combination(self, &[(&a3, c[0]), (&a2, c[1]), (a, c[2])]);
                let y03 = self.matmat(&a3, &inner);
                let left = combination(self, &[(&y03, 1.0), (&a3, c[3]), (&a2, c[4]), (a, c[5])]);
                let right = combination(self, &[(&y03, 1.0), (&a3, c[6]), (&a2, c[7])]);
                let product = self.matmat(&left, &right);
                let y13 = combination(
                    self,
                    &[(&product, 1.0), (&y03, c[8]), (&a3, c[9]), (&a2, c[10])],
                );
                let left =
                    combination(self, &[(&y13, 1.0), (&a3, c[11]), (&a2, c[12]), (a, c[13])]);
                let right = combination(self, &[(&y13, 1.0), (&y03, c[14]), (a, c[15])]);
                let product = self.matmat(&left, &right);
                combination(
                    self,
                    &[
                        (&product, 1.0),
                        (&y13, c[16]),
                        (&y03, c[17]),
                        (&a3, c[18]),
                        (&a2, c[19]),
                        (a, 1.0),
                        (identity, 1.0),
                    ],
                )
            }
            Dtype::F64 => {
                let a6 = self.matmat(&a3, &a3);
                let b1 = combination(
                    self,
                    &[
                        (a, -0.10036558103014462),
                        (&a2, -0.008_029_246_482_411_57),
                        (&a3, -0.00089213849804573),
                    ],
                );
                let b5 = combination(
                    self,
                    &[
                        (&a2, -0.09233646193671186),
                        (&a3, -0.01693649390020817),
                        (&a6, -0.00001400867981820361),
                    ],
                );
                let product = self.matmat(&b1, &b5);
                // Center the factorization at I to preserve exp(0) exactly.
                let r = combination(
                    self,
                    &[
                        (&product, 1.0),
                        (a, -0.06764045190713819),
                        (&a2, 0.06759613017704596),
                        (&a3, 0.02955525704293155),
                        (&a6, -0.00001391802575160607),
                    ],
                );
                let d = combination(
                    self,
                    &[
                        (a, 1.680158138789062),
                        (&a2, 0.05717798464788655),
                        (&a3, -0.006982101224880521),
                        (&a6, 0.00003349750170860705),
                    ],
                );
                let b2 = combination(
                    self,
                    &[
                        (a, 0.3978497494996451),
                        (&a2, 1.367837784604117),
                        (&a3, 0.4982896225253827),
                        (&a6, -0.0006378981945947233),
                    ],
                );
                let left = self.add(&d, &r);
                let product = self.matmat(&left, &r);
                let c = -0.09043168323908106;
                let d0 = -10.967639605296206;
                combination(
                    self,
                    &[
                        (identity, 1.0),
                        (&b2, 1.0),
                        (&d, c),
                        (&r, d0 + 2.0 * c),
                        (&product, 1.0),
                    ],
                )
            }
        }
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
