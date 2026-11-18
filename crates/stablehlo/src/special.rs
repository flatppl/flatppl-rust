//! `special-functions` standard-module calls (§09), which FlatPDL keeps as calls.
//! `erf`, `erfc`, `digamma` and `polygamma` lower to their CHLO ops. `erfcx` has no
//! CHLO op, so it lowers to StableHLO arithmetic that Enzyme differentiates.

use super::*;

#[path = "special/bessel.rs"]
mod bessel;
#[path = "special/probability.rs"]
mod probability;

/// The `special-functions` members this backend lowers, with their arity.
pub(crate) const SPECIAL_FUNCTIONS: [(&str, usize); 25] = [
    ("erf", 1),
    ("erfc", 1),
    ("erfcx", 1),
    ("digamma", 1),
    ("polygamma", 2),
    ("gammainc", 2),
    ("betainc", 3),
    ("bessel_j", 2),
    ("bessel_y", 2),
    ("bessel_i", 2),
    ("bessel_k", 2),
    ("gammaincc", 2),
    ("log_gammainc", 2),
    ("log_gammaincc", 2),
    ("betaincc", 3),
    ("log_betainc", 3),
    ("log_betaincc", 3),
    ("beta", 2),
    ("logbeta", 2),
    ("bessel_i_scaled", 2),
    ("bessel_k_scaled", 2),
    ("log_bessel_i", 2),
    ("log_bessel_k", 2),
    ("log_ndtr", 1),
    ("airy", 1),
];

const ERFCX_SPLIT: f64 = 4.0;
const FRAC_1_SQRT_PI: f64 = 0.5641895835477563;

impl Emitter<'_> {
    /// A static tape bound with a safe, runtime-budgeted prefix. Final
    /// convergence checks reject workspaces too small for the requested rule.
    fn special_loop(
        &mut self,
        maximum: u64,
        limit: &Value,
        initial: &[Value],
        mut step: impl FnMut(&mut Self, &Value, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let requested = self.ceil(limit);
        let rank = shape(&requested.ty).len();
        let requested = self.axes_view(
            &requested,
            Axes {
                batch: 0,
                layers: if rank == 0 { vec![] } else { vec![rank] },
            },
        );
        let requested = self.reduce_max(&requested);
        let count = self.convert(&requested, ElemKind::Int);
        let one = self.int_value_const(1);
        let last = self.sub(&count, &one);
        let output = self.matrix_loop(maximum, 0, initial, |e, index, state| {
            let state: Vec<_> = state
                .iter()
                .zip(initial)
                .map(|(value, initial)| e.axes_view(value, e.axes_of(initial)))
                .collect();
            let active = e.compare("LT", index, &count);
            let safe_index = e.min(index, &last);
            let next = step(e, &safe_index, &state);
            next.into_iter()
                .zip(&state)
                .map(|(next, old)| {
                    let next = e.select(&active, &next, old);
                    e.axes_view(&next, e.axes_of(old))
                })
                .collect()
        });
        let bound = self.scalar(maximum as f64);
        let fits = self.compare("LE", &requested, &bound);
        output
            .iter()
            .zip(initial)
            .map(|(value, initial)| {
                let value = self.axes_view(value, self.axes_of(initial));
                let nan = self.constant_like(f64::NAN, &value);
                self.select(&fits, &value, &nan)
            })
            .collect()
    }
    /// Principal J/I continuation. Complex values retain the backend's pair
    /// representation until a real/imaginary projection consumes them.
    pub(crate) fn lower_bessel_parts(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<(Value, Value), EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "special-functions"
            || !matches!(member.as_str(), "bessel_j" | "bessel_i" | "bessel_i_scaled")
        {
            return None;
        }
        Some((|| {
            let [order, x] = args else {
                return Err(EmitError::at(id, "Bessel J/I require order and argument"));
            };
            let order_value = self.lower_node(*order)?;
            let x_value = self.lower_node(*x)?;
            let mut values = [
                self.typed_axes(*order, order_value),
                self.typed_axes(*x, x_value),
            ];
            let parent = if broadcast {
                Some(self.enter_broadcast(id, &mut values)?)
            } else {
                None
            };
            let (re, im) = self.bessel_principal_parts(&member, &values[0], &values[1]);
            Ok(match parent {
                Some(parent) => {
                    let frame = self.broadcast_frame.clone();
                    let re = self.finish_broadcast(&re, &frame, parent.len());
                    let im = self.finish_broadcast(&im, &frame, parent.len());
                    self.broadcast_frame = parent;
                    (re, im)
                }
                None => (re, im),
            })
        })())
    }
    pub(super) fn lower_special_function_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "special-functions" {
            return None;
        }
        let Some(&(_, arity)) = SPECIAL_FUNCTIONS.iter().find(|(name, _)| *name == member) else {
            return Some(Err(EmitError::at(
                id,
                format!("special-functions.{member} has no StableHLO lowering"),
            )));
        };
        Some(self.emit_special_function_call(id, &member, arity, args, broadcast))
    }

    fn emit_special_function_call(
        &mut self,
        id: NodeId,
        member: &str,
        arity: usize,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        if matches!(self.m.node(id), Node::Call(call) if !call.named.is_empty()) {
            return Err(EmitError::at(
                id,
                "special-functions arguments must be bound by determinize",
            ));
        }
        if args.len() != arity {
            return Err(EmitError::at(
                id,
                format!("special-functions.{member} takes {arity} argument(s)"),
            ));
        }
        let mut values = args
            .iter()
            .map(|&arg| {
                let value = self.lower_node(arg)?;
                Ok(self.typed_axes(arg, value))
            })
            .collect::<Result<Vec<_>, EmitError>>()?;
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let result = match member {
            "erf" => self.chlo_unary("chlo.erf", &values[0]),
            // `1 - erf(x)` cancels to zero in the upper tail.
            "erfc" => self.chlo_unary("chlo.erfc", &values[0]),
            "erfcx" => self.erfcx(&values[0], false),
            "digamma" => self.chlo_unary("chlo.digamma", &values[0]),
            "gammainc" => self.regularized_gamma(&values[0], &values[1]),
            "betainc" => self.regularized_beta(&values[0], &values[1], &values[2]),
            "gammaincc" | "log_gammainc" | "log_gammaincc" => self.regularized_gamma_tail(
                &values[0],
                &values[1],
                member.ends_with("cc"),
                member.starts_with("log_"),
                true,
            ),
            "betaincc" | "log_betainc" | "log_betaincc" => self.regularized_beta_tail(
                &values[0],
                &values[1],
                &values[2],
                member.ends_with("cc"),
                member.starts_with("log_"),
                true,
            ),
            "beta" | "logbeta" => {
                let a = self.convert(&values[0], ElemKind::Real);
                let b = self.convert(&values[1], ElemKind::Real);
                let zero = self.scalar(0.0);
                let positive_a = self.compare("GT", &a, &zero);
                let positive_b = self.compare("GT", &b, &zero);
                let valid = self.and(&positive_a, &positive_b);
                let finite_a = self.special_finite(&a);
                let finite_b = self.special_finite(&b);
                let finite = self.and(&finite_a, &finite_b);
                let valid = self.and(&valid, &finite);
                self.record_runtime_error(&valid, 12);
                let one = self.scalar(1.0);
                let a = self.select(&valid, &a, &one);
                let b = self.select(&valid, &b, &one);
                let value = self.log_beta(&a, &b);
                let value = if member == "beta" {
                    self.exp(&value)
                } else {
                    value
                };
                let nan = self.scalar(f64::NAN);
                self.select(&valid, &value, &nan)
            }
            "bessel_k_scaled" | "log_bessel_i" | "log_bessel_k" => {
                let modified_i = member == "log_bessel_i";
                let scaled = member == "bessel_k_scaled";
                let value = self.bessel_modified_log(&values[0], &values[1], modified_i, scaled);
                if scaled { self.exp(&value) } else { value }
            }
            "log_ndtr" => {
                let masks_nan = self.masks_nan(args[0]);
                self.compact_normal_log_cdf(id, &values[0], masks_nan)
                    .unwrap_or_else(|| self.normal_log_cdf(&values[0], masks_nan))
            }
            "bessel_j" | "bessel_i" | "bessel_i_scaled" => {
                return Err(EmitError::at(
                    id,
                    "complex Bessel J/I result requires real, imag, abs or abs2 projection",
                ));
            }
            "bessel_y" | "bessel_k" => self.bessel_real(member, &values[0], &values[1]),
            "airy" => self.airy_ai(&values[0]),
            // Enzyme differentiates polygamma in `x` only. The order `n` is a §09
            // integer, so it reaches the op as an inactive constant or integer input.
            _ => self.chlo_binary("chlo.polygamma", &values[0], &values[1]),
        };
        Ok(match parent {
            Some(parent) => {
                let frame = self.broadcast_frame.clone();
                let result = self.finish_broadcast(&result, &frame, parent.len());
                self.broadcast_frame = parent;
                result
            }
            None => result,
        })
    }

    fn special_finite(&mut self, x: &Value) -> Value {
        let magnitude = self.abs(x);
        let infinity = self.constant_like(f64::INFINITY, x);
        self.compare("LT", &magnitude, &infinity)
    }

    /// A real value explicitly replaced at NaN does not need a second check.
    fn masks_nan(&self, id: NodeId) -> bool {
        let call = |id, name| {
            let (id, _) = flatppl_core::graph::resolve_ref(self.m, id);
            match self.m.node(id) {
                Node::Call(call) if matches!(call.head, CallHead::Builtin(s) if self.m.resolve(s) == name) => {
                    Some(call)
                }
                _ => None,
            }
        };
        let Some([condition, fallback, value]) = call(id, "ifelse").map(|call| call.args.as_ref())
        else {
            return false;
        };
        let Some([input]) = call(*condition, "isnan").map(|call| call.args.as_ref()) else {
            return false;
        };
        let resolve = |&id| flatppl_core::graph::resolve_ref(self.m, id).0;
        let (input, value) = (resolve(input), resolve(value));
        let same = input == value
            || matches!((self.m.node(input), self.m.node(value)), (Node::Ref(a), Node::Ref(b)) if a == b);
        same && matches!(self.m.node(resolve(fallback)), Node::Lit(Scalar::Real(x)) if !x.is_nan())
    }

    fn normal_log_cdf(&mut self, x: &Value, masks_nan: bool) -> Value {
        let x = self.convert(x, ElemKind::Real);
        let value = if self.restrict_enzyme_compatible
            && self.target == crate::Target::Cpu
            && shape(&x.ty).iter().all(Option::is_some)
        {
            self.normal_log_cdf_with_slope(&x)
        } else {
            self.normal_log_cdf_body(&x, false).0
        };
        if masks_nan {
            return value;
        }
        let not_nan = self.compare("EQ", &x, &x);
        if not_nan.ty == MlirTy::Scalar
            && self.fixed_scalar_at(&not_nan, 0) == Some(Scalar::Bool(true))
        {
            return value;
        }
        self.record_runtime_error(&not_nan, 12);
        let nan = self.constant_like(f64::NAN, &x);
        self.select(&not_nan, &value, &nan)
    }

    pub(super) fn normal_log_cdf_with_slope(&mut self, x: &Value) -> Value {
        let ty = x.ty.render(self.dtype, ElemKind::Real);
        let name = format!("__enzyme_log_ndtr_v1_{}", &ty[7..ty.len() - 1]);
        if !self.solve_helpers.contains_key(&name) {
            let mut helper = self.scratch_emitter();
            let arg = Value {
                ssa: "%x".into(),
                ..x.clone()
            };
            let (value, slope) = helper.normal_log_cdf_body(&arg, true);
            let args = [(arg.ssa, arg.ty, arg.elem)];
            let definition = helper.finish_function(
                &name,
                &args,
                &[&value, &slope.expect("slope requested")],
                true,
            );
            self.solve_helpers.insert(name.clone(), definition);
        }
        let result = self.fresh();
        self.push(&format!(
            "{result}:2 = stablehlo.composite \"enzyme.math.log_ndtr\" {} {{decomposition = @{name}, version = 1 : i32}} : ({ty}) -> ({ty}, {ty})",
            x.ssa,
        ));
        let value = Value {
            ssa: format!("{result}#0"),
            ..x.clone()
        };
        self.remember_axes(&value.ssa, self.axes_of(x));
        self.remember_pointwise(&value.ssa, x, Pointwise::LogNdtr(x.clone()));
        value
    }

    fn normal_log_cdf_body(&mut self, x: &Value, with_slope: bool) -> (Value, Option<Value>) {
        let zero = self.constant_like(0.0, x);
        let negative = self.compare("LT", x, &zero);
        let magnitude = self.abs(x);
        let finite = self.special_finite(x);
        let one = self.constant_like(1.0, x);
        let magnitude = self.select(&finite, &magnitude, &one);

        let inv_sqrt_two = self.scalar(std::f64::consts::FRAC_1_SQRT_2);
        let argument = self.mul(&magnitude, &inv_sqrt_two);
        // Keep XLA from fusing the double-precision erfc expansion with the
        // rest of the interval score and its adjoint into one GPU kernel.
        let split_erfc_fusion =
            self.target == crate::Target::Gpu && matches!(self.dtype, Dtype::F64);
        let (scaled_tail, hazard) = if with_slope {
            let (scaled, hazard) = self.normal_log_cdf_scaled_tail(&argument, &magnitude);
            (scaled, Some(hazard))
        } else {
            (self.erfcx(&argument, split_erfc_fusion), None)
        };
        let log_tail = self.log(&scaled_tail);
        let log_two = self.scalar(std::f64::consts::LN_2);
        let log_tail = self.sub(&log_tail, &log_two);
        let square = self.mul(&argument, &argument);
        let log_tail = self.sub(&log_tail, &square);
        let tail = self.exp(&log_tail);
        let minus_tail = self.neg(&tail);
        let log_complement = self.log1p(&minus_tail);
        let value = self.select(&negative, &log_tail, &log_complement);
        let minus_infinity = self.constant_like(f64::NEG_INFINITY, x);
        let endpoint = self.select(&negative, &minus_infinity, &zero);
        let value = self.select(&finite, &value, &endpoint);
        let slope = hazard.map(|hazard| {
            // Convert the survival hazard to the CDF slope by reusing the
            // tail probability, which is already available from the value.
            let probability = self.sub(&one, &tail);
            let ratio = self.div(&tail, &probability);
            let ratio = self.select(&negative, &one, &ratio);
            let slope = self.mul(&hazard, &ratio);
            self.select(&finite, &slope, &zero)
        });
        (value, slope)
    }

    fn normal_log_cdf_scaled_tail(&mut self, x: &Value, abs_z: &Value) -> (Value, Value) {
        // Fit (x + 1/2) * erfcx(x) on the positive half-line using the
        // Shepherd–Laframboise map t = (x - 3.75)/(x + 3.75). Project
        // 2048 Chebyshev nodes to degrees 11 and 22, then use monomial
        // coefficients for parallel polynomial evaluation.
        // The bounded polynomial needs neither an exponential nor a tail split.
        const F32: [f64; 12] = [
            6.187_563_096_338_5e-1,
            -7.012_032_916_069_327e-2,
            1.793_135_197_012_158e-3,
            4.113_920_999_173_533e-2,
            -5.440_713_507_452_408e-2,
            4.614_538_479_565_338e-2,
            -2.932_342_936_777_644e-2,
            1.420_462_618_205_788e-2,
            -4.922_247_429_833_534e-3,
            8.379_561_089_974_229e-4,
            1.981_538_064_929_362e-4,
            -1.120_583_733_746_959e-4,
        ];
        const F64: [f64; 23] = [
            6.187_563_154_189_137e-1,
            -7.012_029_929_277_282e-2,
            1.792_707_742_735_378e-3,
            4.113_836_924_500_378e-2,
            -5.440_196_507_112e-2,
            4.615_216_058_203_803e-2,
            -2.934_669_928_934_497e-2,
            1.418_113_868_696_314e-2,
            -4.873_289_836_317_03e-3,
            8.778_130_778_583_207e-4,
            1.468_571_728_115_197e-4,
            -1.450_776_608_023_793e-4,
            2.582_329_512_300_962e-5,
            1.119_375_084_495_728e-5,
            -5.719_058_075_668_331e-6,
            -4.895_890_700_778_886e-7,
            8.710_116_694_690_917e-7,
            -2.545_649_480_101_187e-8,
            -1.242_888_959_379_924e-7,
            9.410_866_809_416_574e-9,
            1.598_670_336_777_831e-8,
            -9.805_568_197_344_875e-10,
            -1.310_324_828_046_744e-9,
        ];
        let coefficients = match self.dtype {
            Dtype::F32 => &F32[..],
            Dtype::F64 => &F64[..],
        };
        let shift = self.scalar(3.75);
        let numerator = self.sub(x, &shift);
        let denominator = self.add(x, &shift);
        let mut power = self.div(&numerator, &denominator);
        let mut terms: Vec<_> = coefficients.iter().map(|&c| self.scalar(c)).collect();
        while terms.len() > 1 {
            terms = terms
                .chunks(2)
                .map(|pair| match pair {
                    [a, b] => {
                        let weighted = self.mul(b, &power);
                        self.add(a, &weighted)
                    }
                    [a] => a.clone(),
                    _ => unreachable!(),
                })
                .collect();
            if terms.len() > 1 {
                power = self.mul(&power, &power);
            }
        }
        let normalized = &terms[0];
        let half = self.scalar(0.5);
        let scale = self.add(x, &half);
        let scaled = self.div(normalized, &scale);

        let shift = self.scalar(std::f64::consts::FRAC_1_SQRT_2);
        let scale = self.add(abs_z, &shift);
        let inverse_sqrt_pi = self.scalar(FRAC_1_SQRT_PI);
        let ratio = self.div(&inverse_sqrt_pi, normalized);
        let hazard = self.mul(&scale, &ratio);
        // Beyond this point, the hazard correction is below half an ulp.
        let cutoff = self.scalar(match self.dtype {
            Dtype::F32 => 4096.0,
            Dtype::F64 => 134_217_728.0,
        });
        let far = self.compare("GE", abs_z, &cutoff);
        (scaled, self.select(&far, abs_z, &hazard))
    }

    /// Keep fixed infinite endpoints out of the CDF and its adjoint.
    /// Restore the original lane order by indexing the compact result and
    /// the two constant endpoint values. Only compact rows common to the
    /// whole batch, so rebinding or batching dynamic inputs stays valid.
    fn compact_normal_log_cdf(&mut self, id: NodeId, x: &Value, masks_nan: bool) -> Option<Value> {
        if self.target != crate::Target::Cpu {
            return None;
        }
        let (width, prefix) = shape(&x.ty).split_last()?;
        let width = usize::try_from((*width)?).ok()?;
        let batch = prefix.iter().try_fold(1usize, |count, n| {
            count.checked_mul(usize::try_from((*n)?).ok()?)
        })?;
        if !(64..=constants::MAX_FOLDED_ELEMENTS).contains(&width) || !(1..=32).contains(&batch) {
            return None;
        }
        let endpoint = |e: &Self, i| match e.fixed_scalar_at(x, i) {
            Some(Scalar::Real(v)) if v.is_infinite() => Some(v.is_sign_negative()),
            _ => None,
        };
        let mut active = Vec::new();
        let mut layout = Vec::with_capacity(width);
        for i in 0..width {
            let fixed = endpoint(self, i)
                .filter(|sign| (1..batch).all(|b| endpoint(self, b * width + i) == Some(*sign)));
            layout.push(fixed);
            if fixed.is_none() {
                active.push(i as i64);
            }
        }
        if active.len() > width * 3 / 4 {
            return None;
        }
        let n = active.len() as i64;
        let indices = |e: &mut Self, values: Vec<i64>| {
            let ty = MlirTy::Ranked(vec![Some(values.len() as u64)]);
            e.folded_constant(
                values.into_iter().map(Scalar::Int).collect(),
                ty,
                Axes::default(),
            )
        };
        let original = self.axes_of(x);
        let vector_axes = Axes {
            batch: prefix.len(),
            layers: vec![1],
        };
        let rows = self.axes_view(x, vector_axes.clone());
        let mut parts = Vec::with_capacity(3);
        if n != 0 {
            let keep = indices(self, active)?;
            let rows = self.gather_axis(&rows, &keep, 0, 0);
            parts.push(self.normal_log_cdf(&rows, masks_nan));
        }
        let mut tail_shape = prefix.to_vec();
        tail_shape.push(Some(1));
        for value in [0.0, f64::NEG_INFINITY] {
            let tail = self.constant(value, MlirTy::Ranked(tail_shape.clone()));
            parts.push(self.axes_view(&tail, vector_axes.clone()));
        }
        let merged = self.concatenate(id, &parts).ok()?;
        let mut active_slot = 0;
        let layout = layout
            .into_iter()
            .map(|endpoint| match endpoint {
                Some(negative) => n + i64::from(negative),
                None => {
                    let slot = active_slot;
                    active_slot += 1;
                    slot
                }
            })
            .collect();
        let restore = indices(self, layout)?;
        let restored = self.gather_axis(&merged, &restore, 0, 0);
        Some(self.axes_view(&restored, original))
    }

    /// Use the direct product near zero and Cody's rational tail beyond 4.
    /// Clamp inactive inputs so they cannot overflow in the derivative.
    fn erfcx(&mut self, x: &Value, split_erfc_fusion: bool) -> Value {
        let x = &self.convert(x, ElemKind::Real);
        let split = self.constant_like(ERFCX_SPLIT, x);
        let small = self.compare("LT", x, &split);

        let zero = self.constant_like(0.0, x);
        let near = self.select(&small, x, &zero);
        let square = self.mul(&near, &near);
        let scale = self.exp(&square);
        let tail = self.chlo_unary("chlo.erfc", &near);
        let tail = if split_erfc_fusion {
            self.unary("stablehlo.optimization_barrier", &tail)
        } else {
            tail
        };
        let product = self.mul(&scale, &tail);

        // Cody (1969), Math. Comp. 23, 631–638. The coefficients are the
        // large-argument P and Q from https://netlib.org/specfun/erf, in
        // ascending powers of 1/x². Form 1/x first to avoid squaring a large x.
        // A distinct placeholder avoids folding this select to max(x, 4),
        // which can send the tangent at the split to the constant.
        let far_placeholder = self.scalar(5.0);
        let far = self.select(&small, &far_placeholder, x);
        let one = self.scalar(1.0);
        let inverse = self.div(&one, &far);
        let t = self.mul(&inverse, &inverse);
        let p = crate::registry::horner(
            self,
            &t,
            &[
                6.587_491_615_298_378e-4,
                1.608_378_514_874_228e-2,
                1.257_817_261_112_292_5e-1,
                3.603_448_999_498_044_4e-1,
                3.053_266_349_612_323_4e-1,
                1.631_538_713_730_209_8e-2,
            ],
        );
        let q = crate::registry::horner(
            self,
            &t,
            &[
                2.335_204_976_268_691_8e-3,
                6.051_834_131_244_132e-2,
                5.279_051_029_514_284e-1,
                1.872_952_849_923_460_5,
                2.568_520_192_289_822,
                1.0,
            ],
        );
        let p = self.mul(&t, &p);
        let correction = self.div(&p, &q);
        let leading = self.scalar(FRAC_1_SQRT_PI);
        let rational = self.sub(&leading, &correction);
        let tail = self.mul(&inverse, &rational);
        self.select(&small, &product, &tail)
    }
}
