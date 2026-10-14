//! Full IID log densities from fixed observations and shared scalar parameters.
//! Read constants after ABI binding, so a promoted fixed input stays live.

use super::*;
use flatppl_core::Call;

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    fn add(&mut self, x: f64) {
        let next = self.value + x;
        self.correction += if self.value.abs() >= x.abs() {
            (self.value - next) + x
        } else {
            (x - next) + self.value
        };
        self.value = next;
    }

    fn total(&self) -> f64 {
        self.value + self.correction
    }
}

impl Emitter<'_> {
    fn density_call(&self, mut id: NodeId, name: &str) -> Option<Call> {
        for _ in 0..64 {
            // A bound expression supplies its caller's value, not its source RHS.
            if self.memo.contains_key(&id) {
                return None;
            }
            match self.m.node(id) {
                Node::Ref(Ref {
                    ns: RefNs::SelfMod,
                    name,
                }) => {
                    id = self.m.binding(self.m.binding_by_name(*name)?).rhs;
                }
                Node::Call(call) if matches!(call.head, CallHead::Builtin(s) if self.m.resolve(s) == name) =>
                {
                    return Some(call.clone());
                }
                _ => return None,
            }
        }
        None
    }

    fn density_scalar(&mut self, id: NodeId) -> Result<Option<Value>, EmitError> {
        let value = self.lower_node(id)?;
        let mut axes = self.axes_of(&value);
        let dims = shape(&value.ty);
        if dims[axes.batch..].iter().all(|&n| n == Some(1)) {
            let ty = batching::tensor(dims[..axes.batch].to_vec());
            axes.layers.clear();
            return Ok(Some(self.reshape_axes(&value, ty, axes)));
        }
        // A fixed splat is shared even when spelled as a full parameter vector.
        if axes.batch == 0
            && let Some(data) = self.constants.get(&value.ssa)
            && let Some(first) = data.first()
            && data.iter().all(|x| x == first)
        {
            return Ok(Some(match first.clone() {
                Scalar::Real(x) => self.scalar(x),
                Scalar::Int(x) => self.int_value_const(x),
                _ => return Ok(None),
            }));
        }
        Ok(None)
    }

    fn summary_scalar(&mut self, x: f64) -> Option<Value> {
        let rounded = match self.dtype {
            Dtype::F32 => (x as f32) as f64,
            Dtype::F64 => x,
        };
        (rounded.is_finite() && (x == 0.0 || rounded != 0.0)).then(|| self.scalar(rounded))
    }

    pub(crate) fn reduce_fixed_likelihood(
        &mut self,
        arg: NodeId,
    ) -> Result<Option<Value>, EmitError> {
        let Some(call) = self.density_call(arg, "broadcast") else {
            return Ok(None);
        };
        let [head, kernel, input, observed] = call.args.as_ref() else {
            return Ok(None);
        };
        if !call.named.is_empty()
            || !matches!(self.m.node(*head), Node::Const(s) if self.resolve(*s) == "builtin_logdensityof")
        {
            return Ok(None);
        }
        let ctor = match self.m.node(*kernel) {
            Node::Const(s) => self.resolve(*s),
            _ => return Ok(None),
        };
        let names: &[&str] = match ctor {
            "Normal" => &["mu", "sigma"],
            "Gamma" => &["shape", "rate"],
            "Beta" => &["alpha", "beta"],
            "Exponential" | "Poisson" => &["rate"],
            "Binomial" => &["n", "p"],
            _ => return Ok(None),
        };
        let ctor = ctor.to_owned();
        let Some(input) = self.density_call(*input, "broadcast") else {
            return Ok(None);
        };
        if !matches!(input.args.as_ref(), [head] if matches!(self.m.node(*head), Node::Const(s) if self.resolve(*s) == "record"))
            || input.named.len() != names.len()
        {
            return Ok(None);
        }
        let fields = names
            .iter()
            .map(|name| {
                input.named.iter().find_map(|field| {
                    (field.kind == NamedKind::Kwarg && self.resolve(field.name) == *name)
                        .then_some(field.value)
                })
            })
            .collect::<Option<Vec<_>>>();
        let Some(fields) = fields else {
            return Ok(None);
        };

        let observations = self.lower_node(*observed)?;
        // Continuous primitives can subtract/multiply integer operands before
        // widening them. Floating summaries would change that rounding order.
        if observations.elem != ElemKind::Real
            && matches!(ctor.as_str(), "Normal" | "Gamma" | "Exponential")
        {
            return Ok(None);
        }
        let [Some(count)] = shape(&observations.ty) else {
            return Ok(None);
        };
        if *count == 0 || self.batch_rank(&observations) != 0 {
            return Ok(None);
        }
        let Some(data) = self.constants.get(&observations.ssa).cloned() else {
            return Ok(None);
        };
        let count = *count as usize;
        let exact_limit = match self.dtype {
            Dtype::F32 => 1_u64 << 24,
            Dtype::F64 => 1_u64 << 53,
        };
        if count as u64 > exact_limit || (data.len() != 1 && data.len() != count) {
            return Ok(None);
        }
        let real = |x: &Scalar| match x {
            Scalar::Real(x) => Some(*x),
            Scalar::Int(x) => Some(*x as f64),
            _ => None,
        };
        let values = || data.iter();
        let repetitions = (count / data.len()) as f64;
        let mut sum = Sum::default();
        let mut log_sum = Sum::default();
        let mut other_sum = Sum::default();
        for x in values() {
            let Some(x) = real(x).filter(|x| x.is_finite()) else {
                return Ok(None);
            };
            match ctor.as_str() {
                "Gamma" if x <= 0.0 => return Ok(None),
                "Beta" if !(0.0 < x && x < 1.0) => return Ok(None),
                "Exponential" if x < 0.0 => return Ok(None),
                "Poisson" if !(0.0..64.0).contains(&x) || x.fract() != 0.0 => return Ok(None),
                "Binomial" if x < 0.0 || x.fract() != 0.0 => return Ok(None),
                _ => {}
            }
            sum.add(x);
            if matches!(ctor.as_str(), "Gamma" | "Beta") {
                log_sum.add(x.ln());
            }
            if ctor == "Beta" {
                other_sum.add((-x).ln_1p());
            } else if ctor == "Poisson" {
                other_sum.add(libm::lgamma(x + 1.0));
            }
        }
        let mut params = Vec::with_capacity(fields.len());
        for field in fields {
            let Some(value) = self.density_scalar(field)? else {
                return Ok(None);
            };
            params.push(value);
        }
        let Some(n) = self.summary_scalar(count as f64) else {
            return Ok(None);
        };
        let sum = sum.total() * repetitions;
        let Some(total) = self.summary_scalar(sum) else {
            return Ok(None);
        };
        let result = match ctor.as_str() {
            "Normal" => {
                let center = real(&data[0]).unwrap();
                let mut offsets = Sum::default();
                for x in values() {
                    offsets.add(real(x).unwrap() - center);
                }
                let offset = offsets.total() / data.len() as f64;
                // Keep a representable center near the mean. A residual sum
                // corrects its rounding without adding a tiny offset to it in
                // runtime IR, where reassociation could round that offset away.
                let center = match self.dtype {
                    Dtype::F32 => ((center + offset) as f32) as f64,
                    Dtype::F64 => center + offset,
                };
                let mut residuals = Sum::default();
                let mut squares = Sum::default();
                for x in values() {
                    let residual = real(x).unwrap() - center;
                    let square = residual * residual;
                    if residual != 0.0 && square == 0.0 {
                        return Ok(None);
                    }
                    residuals.add(residual);
                    squares.add(square);
                }
                let (Some(center), Some(residual_sum), Some(spread)) = (
                    self.summary_scalar(center),
                    self.summary_scalar(residuals.total() * repetitions),
                    self.summary_scalar((squares.total() * repetitions).sqrt()),
                ) else {
                    return Ok(None);
                };
                let difference = self.sub(&center, &params[0]);
                let z = self.div(&difference, &params[1]);
                let z2 = self.mul(&z, &z);
                let mean_quad = self.mul(&n, &z2);
                let residual_sum = self.div(&residual_sum, &params[1]);
                let cross = self.mul(&z, &residual_sum);
                let two = self.scalar(2.0);
                let cross = self.mul(&two, &cross);
                let spread = self.div(&spread, &params[1]);
                let variance_quad = self.mul(&spread, &spread);
                let quad = self.add(&mean_quad, &cross);
                let quad = self.add(&quad, &variance_quad);
                let half = self.scalar(-0.5);
                let quad = self.mul(&half, &quad);
                let log_scale = self.log(&params[1]);
                let constant = self.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
                let normalizer = self.add(&log_scale, &constant);
                let normalizer = self.mul(&n, &normalizer);
                self.sub(&quad, &normalizer)
            }
            "Gamma" => {
                let Some(log_total) = self.summary_scalar(log_sum.total() * repetitions) else {
                    return Ok(None);
                };
                let log_rate = self.log(&params[1]);
                let shape_log_rate = self.mul(&params[0], &log_rate);
                let lgamma = self.lgamma(&params[0]);
                let normalizer = self.sub(&shape_log_rate, &lgamma);
                let normalizer = self.mul(&n, &normalizer);
                let one = self.scalar(1.0);
                let shape_minus_one = self.sub(&params[0], &one);
                let data_term = self.mul(&shape_minus_one, &log_total);
                let score = self.add(&normalizer, &data_term);
                let rate_sum = self.mul(&params[1], &total);
                self.sub(&score, &rate_sum)
            }
            "Beta" => {
                let (Some(log_total), Some(complement_total)) = (
                    self.summary_scalar(log_sum.total() * repetitions),
                    self.summary_scalar(other_sum.total() * repetitions),
                ) else {
                    return Ok(None);
                };
                let one = self.scalar(1.0);
                let a_minus_one = self.sub(&params[0], &one);
                let a_term = self.mul(&a_minus_one, &log_total);
                let b_minus_one = self.sub(&params[1], &one);
                let b_term = self.mul(&b_minus_one, &complement_total);
                let data_term = self.add(&a_term, &b_term);
                let a_gamma = self.lgamma(&params[0]);
                let b_gamma = self.lgamma(&params[1]);
                let both = self.add(&params[0], &params[1]);
                let both_gamma = self.lgamma(&both);
                let gamma_sum = self.add(&a_gamma, &b_gamma);
                let normalizer = self.sub(&gamma_sum, &both_gamma);
                let normalizer = self.mul(&n, &normalizer);
                self.sub(&data_term, &normalizer)
            }
            "Exponential" => {
                let log_rate = self.log(&params[0]);
                let normalizer = self.mul(&n, &log_rate);
                let rate_sum = self.mul(&params[0], &total);
                self.sub(&normalizer, &rate_sum)
            }
            "Poisson" => {
                let Some(factorials) = self.summary_scalar(other_sum.total() * repetitions) else {
                    return Ok(None);
                };
                let zero = self.scalar(0.0);
                let one = self.scalar(1.0);
                let zero_rate = self.compare("EQ", &params[0], &zero);
                let safe_rate = self.select(&zero_rate, &one, &params[0]);
                let log_rate = self.log(&safe_rate);
                let counts_term = self.mul(&total, &log_rate);
                let rate_sum = self.mul(&n, &params[0]);
                let score = self.sub(&counts_term, &rate_sum);
                let score = self.sub(&score, &factorials);
                if sum == 0.0 {
                    score
                } else {
                    let inf = self.inf_like(&score);
                    let outside = self.neg(&inf);
                    self.select(&zero_rate, &outside, &score)
                }
            }
            "Binomial" => {
                let Some(trials) = self
                    .constants
                    .get(&params[0].ssa)
                    .and_then(|v| v.first())
                    .and_then(real)
                else {
                    return Ok(None);
                };
                let exact_limit = match self.dtype {
                    Dtype::F32 => (1_u64 << 24) as f64,
                    Dtype::F64 => (1_u64 << 53) as f64,
                };
                if trials < 0.0 || trials > exact_limit || trials.fract() != 0.0 {
                    return Ok(None);
                }
                let mut coefficients = Sum::default();
                for x in values() {
                    let x = real(x).unwrap();
                    if x > trials {
                        return Ok(None);
                    }
                    coefficients.add(
                        libm::lgamma(trials + 1.0)
                            - libm::lgamma(x + 1.0)
                            - libm::lgamma(trials - x + 1.0),
                    );
                }
                let (Some(coefficients), Some(failures)) = (
                    self.summary_scalar(coefficients.total() * repetitions),
                    self.summary_scalar(count as f64 * trials - sum),
                ) else {
                    return Ok(None);
                };
                let successes = if sum == 0.0 {
                    self.scalar(0.0)
                } else {
                    let log_prob = self.log(&params[1]);
                    self.mul(&total, &log_prob)
                };
                let failures = if count as f64 * trials == sum {
                    self.scalar(0.0)
                } else {
                    let neg_prob = self.neg(&params[1]);
                    let log_complement = self.log1p(&neg_prob);
                    self.mul(&failures, &log_complement)
                };
                let score = self.add(&coefficients, &successes);
                self.add(&score, &failures)
            }
            _ => unreachable!(),
        };
        Ok(Some(result))
    }
}
