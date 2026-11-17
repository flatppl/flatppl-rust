//! Regularized incomplete gamma and beta ratios. DLMF 8.7.1, 8.9.2, 8.17.22.
//! Input-derived trip counts retain reverse AD. Converged lanes stop accumulating, and
//! nonconvergence returns NaN rather than a silently truncated approximation.

use super::*;

impl Emitter<'_> {
    /// DLMF 8.12.3–10. Two uniform Temme terms suffice above 10^6;
    /// Taylor forms remove the removable singularities at x=a.
    fn temme_gamma_log(&mut self, a: &Value, x: &Value, upper: bool) -> Value {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let half = self.scalar(0.5);
        let difference = self.sub(x, a);
        let delta = self.div(&difference, a);
        let minus_half = self.scalar(-0.5);
        let bounded = self.max(&delta, &minus_half);
        let bounded = self.min(&bounded, &half);
        let magnitude = self.abs(&bounded);
        let threshold = self.scalar(0.125);
        let small = self.compare("LT", &magnitude, &threshold);
        let safe_delta = self.select(&small, &zero, &bounded);
        let logarithm = self.special_log1pmx(&safe_delta);
        let square = self.mul(&safe_delta, &safe_delta);
        let square = self.select(&small, &one, &square);
        let minus_two = self.scalar(-2.0);
        let direct_ratio = self.mul(&minus_two, &logarithm);
        let direct_ratio = self.div(&direct_ratio, &square);
        let coefficients: Vec<_> = (0..23)
            .map(|k| {
                if k % 2 == 0 {
                    2.0 / (k + 2) as f64
                } else {
                    -2.0 / (k + 2) as f64
                }
            })
            .collect();
        let ratio = crate::registry::horner(self, &bounded, &coefficients);
        let ratio = self.select(&small, &ratio, &direct_ratio);
        let ratio = self.sqrt(&ratio);
        let eta = self.mul(&bounded, &ratio);
        let near_threshold = self.scalar(0.01);
        let near = self.compare("LT", &magnitude, &near_threshold);
        let safe_d = self.select(&near, &one, &bounded);
        let safe_eta = self.select(&near, &one, &eta);
        let inverse_d = self.div(&one, &safe_d);
        let inverse_eta = self.div(&one, &safe_eta);
        let c0_direct = self.sub(&inverse_d, &inverse_eta);
        let d2 = self.mul(&inverse_d, &inverse_d);
        let d3 = self.mul(&d2, &inverse_d);
        let eta2 = self.mul(&inverse_eta, &inverse_eta);
        let eta3 = self.mul(&eta2, &inverse_eta);
        let c1_direct = self.sub(&eta3, &d3);
        let c1_direct = self.sub(&c1_direct, &d2);
        let one_twelfth = self.scalar(1.0 / 12.0);
        let correction = self.mul(&one_twelfth, &inverse_d);
        let c1_direct = self.sub(&c1_direct, &correction);
        let c0 = crate::registry::horner(
            self,
            &bounded,
            &[
                -1.0 / 3.0,
                1.0 / 12.0,
                -23.0 / 540.0,
                353.0 / 12960.0,
                -589.0 / 30240.0,
                81083.0 / 5443200.0,
                -7783.0 / 653184.0,
                514303.0 / 52254720.0,
                -646245559.0 / 77598259200.0,
            ],
        );
        let c1 = crate::registry::horner(
            self,
            &bounded,
            &[
                -1.0 / 540.0,
                -1.0 / 288.0,
                23.0 / 6048.0,
                -3733.0 / 1088640.0,
                3253.0 / 1088640.0,
                -135719.0 / 52254720.0,
                176215213.0 / 77598259200.0,
            ],
        );
        let c0 = self.select(&near, &c0, &c0_direct);
        let c1 = self.select(&near, &c1, &c1_direct);
        let c1 = self.div(&c1, a);
        let coefficient = self.add(&c0, &c1);
        let loga = self.log(a);
        let log_two_pi = self.scalar((2.0 * std::f64::consts::PI).ln());
        let log_scale = self.add(&loga, &log_two_pi);
        let log_scale = self.mul(&log_scale, &half);
        let half_a = self.mul(a, &half);
        let root = self.sqrt(&half_a);
        let argument = self.mul(&eta, &root);
        let magnitude = self.abs(&argument);
        let leading = self.erfcx(&magnitude);
        let leading = self.mul(&leading, &half);
        let log_scale = self.neg(&log_scale);
        let scale = self.exp(&log_scale);
        let correction = self.mul(&scale, &coefficient);
        let positive_eta = self.compare("GE", &eta, &zero);
        let negative_correction = self.neg(&correction);
        let correction = self.select(&positive_eta, &correction, &negative_correction);
        let scaled_tail = self.add(&leading, &correction);
        let log_tail = self.log(&scaled_tail);
        let square = self.mul(&argument, &argument);
        let log_tail = self.sub(&log_tail, &square);
        let other = self.special_log1mexp(&log_tail);
        if upper {
            self.select(&positive_eta, &log_tail, &other)
        } else {
            self.select(&positive_eta, &other, &log_tail)
        }
    }
    /// a log(x) - x - log Gamma(a), with the large-shape cancellation
    /// removed analytically near x=a.
    pub(crate) fn gamma_log_prefactor(&mut self, a: &Value, x: &Value) -> Value {
        let logx = self.log(x);
        let power = self.mul(a, &logx);
        let gamma = self.lgamma(a);
        let direct = self.sub(&power, x);
        let direct = self.sub(&direct, &gamma);
        let difference = self.sub(x, a);
        let delta = self.div(&difference, a);
        let magnitude = self.abs(&delta);
        let half = self.scalar(0.5);
        let nearby = self.compare("LT", &magnitude, &half);
        let eight = self.scalar(8.0);
        let large = self.compare("GE", a, &eight);
        let stable = self.and(&nearby, &large);
        let zero = self.scalar(0.0);
        let safe_delta = self.select(&stable, &delta, &zero);
        let logarithm = self.special_log1pmx(&safe_delta);
        let logarithm = self.mul(a, &logarithm);
        let loga = self.log(a);
        let half_loga = self.mul(&half, &loga);
        let constant = self.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
        let remainder = self.stirling_remainder(a);
        let corrected = self.add(&logarithm, &half_loga);
        let corrected = self.sub(&corrected, &constant);
        let corrected = self.sub(&corrected, &remainder);
        self.select(&stable, &corrected, &direct)
    }
    pub(crate) fn special_log1pmx(&mut self, delta: &Value) -> Value {
        let magnitude = self.abs(delta);
        let threshold = self.scalar(0.125);
        let small = self.compare("LT", &magnitude, &threshold);
        let zero = self.scalar(0.0);
        let safe_small = self.select(&small, delta, &zero);
        let mut polynomial = self.scalar(-1.0 / 24.0);
        for k in (2..24).rev() {
            polynomial = self.mul(&polynomial, &safe_small);
            let coefficient = self.scalar(if k % 2 == 0 {
                -1.0 / k as f64
            } else {
                1.0 / k as f64
            });
            polynomial = self.add(&polynomial, &coefficient);
        }
        let square = self.mul(&safe_small, &safe_small);
        let polynomial = self.mul(&polynomial, &square);
        let logarithm = self.log1p(delta);
        let logarithm = self.sub(&logarithm, delta);
        self.select(&small, &polynomial, &logarithm)
    }

    /// log(1-exp(x)) for x≤0, without cancellation near zero.
    fn special_log1mexp(&mut self, x: &Value) -> Value {
        let split = self.constant_like(-std::f64::consts::LN_2, x);
        let far = self.compare("LT", x, &split);
        let safe_far = self.select(&far, x, &split);
        let exponential = self.exp(&safe_far);
        let negative = self.neg(&exponential);
        let far_value = self.log1p(&negative);
        let safe_near = self.select(&far, &split, x);
        let difference = self.expm1(&safe_near);
        let difference = self.neg(&difference);
        let near_value = self.log(&difference);
        self.select(&far, &far_value, &near_value)
    }

    /// DLMF 5.7.3, retaining log Gamma(1+a)/a when 1+a rounds to one.
    pub(super) fn special_log_gamma1p_over_a(&mut self, a: &Value) -> Value {
        let threshold = self.scalar(0.125);
        let small = self.compare("LT", a, &threshold);
        let safe = self.max(a, &threshold);
        let one = self.scalar(1.0);
        let argument = self.add(&one, &safe);
        let gamma = self.lgamma(&argument);
        let direct = self.div(&gamma, &safe);
        let polynomial = crate::registry::horner(
            self,
            a,
            &[
                -std::f64::consts::EULER_GAMMA,
                0.8224670334241132,
                -0.40068563438653143,
                0.27058080842778455,
                -0.20738555102867399,
                0.1695571769974082,
                -0.144_049_896_768_846_1,
                0.12550966952474304,
                -0.11133426586956469,
                0.100_099_457_512_781_8,
                -0.09095401714582904,
                0.083353840546109,
                -0.0769325164113522,
                0.07143294629536134,
                -0.06666870588242047,
                0.06250095514121304,
            ],
        );
        self.select(&small, &polynomial, &direct)
    }

    /// expm1(x)/x or log1p(x)/x, including its derivative at zero.
    fn special_relative(&mut self, x: &Value, exponential: bool) -> Value {
        let threshold = self.scalar(1e-4);
        let magnitude = self.abs(x);
        let small = self.compare("LT", &magnitude, &threshold);
        let one = self.scalar(1.0);
        let safe = self.select(&small, &one, x);
        let numerator = if exponential {
            self.expm1(&safe)
        } else {
            self.log1p(&safe)
        };
        let direct = self.div(&numerator, &safe);
        let coefficients = if exponential {
            [1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0]
        } else {
            [1.0, -0.5, 1.0 / 3.0, -0.25, 0.2]
        };
        let zero = self.scalar(0.0);
        let safe_small = self.select(&small, x, &zero);
        let polynomial = crate::registry::horner(self, &safe_small, &coefficients);
        self.select(&small, &polynomial, &direct)
    }

    fn small_gamma_logs(&mut self, a: &Value, x: &Value) -> (Value, Value) {
        let one = self.constant_like(1.0, x);
        let zero = self.constant_like(0.0, x);
        let axes = self.axes_of(&one);
        let state = self.matrix_loop(32, 0, &[one.clone(), zero], |e, index, state| {
            let state: Vec<_> = state.iter().map(|v| e.axes_view(v, axes.clone())).collect();
            let n = e.convert(index, ElemKind::Real);
            let n = e.add(&n, &one);
            let negative_x = e.neg(x);
            let factor = e.div(&negative_x, &n);
            let term = e.mul(&state[0], &factor);
            let denominator = e.add(a, &n);
            let addend = e.div(&term, &denominator);
            let sum = e.add(&state[1], &addend);
            vec![term, sum]
        });
        let sum = self.axes_view(&state[1], axes);
        let logx = self.log(x);
        let gamma = self.special_log_gamma1p_over_a(a);
        let coefficient = self.sub(&logx, &gamma);
        let logarithm = self.mul(a, &coefficient);
        let relative = self.special_relative(&logarithm, true);
        let left = self.mul(&relative, &coefficient);
        let left = self.neg(&left);
        let scale = self.exp(&logarithm);
        let right = self.mul(&scale, &sum);
        let q_over_a = self.sub(&left, &right);
        let log_q = self.log(&q_over_a);
        let loga = self.log(a);
        let log_q = self.add(&loga, &log_q);
        let correction = self.mul(a, &sum);
        let correction = self.log1p(&correction);
        let log_p = self.add(&logarithm, &correction);
        let split = self.scalar(-std::f64::consts::LN_2);
        let small_p = self.compare("LT", &log_p, &split);
        let complement = self.special_log1mexp(&log_p);
        let log_q = self.select(&small_p, &complement, &log_q);
        (log_p, log_q)
    }

    /// DLMF 8.17.7 with its leading one excluded from the series. The
    /// log-gamma increment is factored by a before it can round away Q.
    fn small_beta_logs(&mut self, a: &Value, b: &Value, x: &Value) -> (Value, Value, Value) {
        let one = self.constant_like(1.0, x);
        let zero = self.constant_like(0.0, x);
        let shifted = self.compare("LT", b, &one);
        let bp1 = self.add(b, &one);
        let argument = self.select(&shifted, &bp1, b);
        let mut difference = self.scalar(0.0);
        for (n, denominator) in [
            (5.0, 720.0),
            (4.0, 120.0),
            (3.0, 24.0),
            (2.0, 6.0),
            (1.0, 2.0),
        ] {
            difference = self.mul(&difference, a);
            let order = self.scalar(n);
            let derivative = self.chlo_binary("chlo.polygamma", &order, &argument);
            let divisor = self.scalar(denominator);
            let derivative = self.div(&derivative, &divisor);
            difference = self.add(&difference, &derivative);
        }
        difference = self.mul(&difference, a);
        let psi = self.chlo_unary("chlo.digamma", &argument);
        difference = self.add(&difference, &psi);
        let gamma = self.special_log_gamma1p_over_a(a);
        let normalization = self.sub(&gamma, &difference);
        let largest = self.max(a, b);
        let ratio = self.div(a, &largest);
        let log_ratio_near = self.log1p(&ratio);
        let inverse = self.div(b, &largest);
        let correction = self.log1p(&inverse);
        let loga = self.log(a);
        let logb = self.log(b);
        let log_ratio_far = self.sub(&loga, &logb);
        let log_ratio_far = self.add(&log_ratio_far, &correction);
        let ordered = self.compare("LE", a, b);
        let log_ratio = self.select(&ordered, &log_ratio_near, &log_ratio_far);
        let log_ratio = self.select(&shifted, &log_ratio, &zero);
        let tolerance = self.scalar(self.ratio_tolerance());
        let budget = self.constant_like(256.0, x);
        let state = self.special_loop(
            256,
            &budget,
            &[one.clone(), zero.clone(), zero.clone()],
            |e, index, state| {
                let n = e.convert(index, ElemKind::Real);
                let n = e.add(&n, &one);
                let factor = e.sub(&n, b);
                let factor = e.mul(&factor, x);
                let factor = e.div(&factor, &n);
                let term = e.mul(&state[0], &factor);
                let denominator = e.add(a, &n);
                let addend = e.div(&term, &denominator);
                let sum = e.add(&state[1], &addend);
                let magnitude = e.abs(&addend);
                let scale = e.abs(&sum);
                let bound = e.mul(&tolerance, &scale);
                let converged = e.compare("LE", &magnitude, &bound);
                let done = e.compare("NE", &state[2], &zero);
                let keep = e.or(&converged, &done);
                let sum = e.select(&done, &state[1], &sum);
                let term = e.select(&keep, &zero, &term);
                let flag = e.convert(&keep, ElemKind::Real);
                vec![term, sum, flag]
            },
        );
        let z = self.mul(a, &state[1]);
        let relative = self.special_relative(&z, false);
        let correction = self.mul(&state[1], &relative);
        let logx = self.log(x);
        let base = self.sub(&logx, &normalization);
        let base = self.add(&base, &correction);
        let log_p = self.mul(a, &base);
        let log_p = self.sub(&log_p, &log_ratio);
        let ratio_over_a = self.div(&log_ratio, a);
        let coefficient = self.sub(&base, &ratio_over_a);
        let negative = self.neg(&coefficient);
        let log_coefficient = self.log(&negative);
        let relative = self.special_relative(&log_p, true);
        let log_relative = self.log(&relative);
        let factored = self.add(&loga, &log_coefficient);
        let factored = self.add(&factored, &log_relative);
        let split = self.scalar(-std::f64::consts::LN_2);
        let small_p = self.compare("LT", &log_p, &split);
        let finite = self.special_finite(&coefficient);
        let finite = self.not(&finite);
        let use_complement = self.or(&small_p, &finite);
        let complement = self.special_log1mexp(&log_p);
        let log_q = self.select(&use_complement, &complement, &factored);
        (log_p, log_q, state[2].clone())
    }

    fn beta_log_prefactor(&mut self, a: &Value, b: &Value, x: &Value) -> Value {
        let logx = self.log(x);
        let negative = self.neg(x);
        let log_complement = self.log1p(&negative);
        let left = self.mul(a, &logx);
        let right = self.mul(b, &log_complement);
        let log_beta = self.log_beta(a, b);
        let direct = self.add(&left, &right);
        let direct = self.sub(&direct, &log_beta);
        let sum = self.add(a, b);
        let p = self.div(a, &sum);
        let q = self.div(b, &sum);
        let delta = self.sub(x, &p);
        let first = self.div(&delta, &p);
        let negative_delta = self.neg(&delta);
        let second = self.div(&negative_delta, &q);
        let half = self.scalar(0.5);
        let eight = self.scalar(8.0);
        let a_large = self.compare("GE", a, &eight);
        let b_large = self.compare("GE", b, &eight);
        let first_magnitude = self.abs(&first);
        let second_magnitude = self.abs(&second);
        let first_near = self.compare("LT", &first_magnitude, &half);
        let second_near = self.compare("LT", &second_magnitude, &half);
        let large = self.and(&a_large, &b_large);
        let near = self.and(&first_near, &second_near);
        let stable = self.and(&large, &near);
        let zero = self.scalar(0.0);
        let first = self.select(&stable, &first, &zero);
        let second = self.select(&stable, &second, &zero);
        let first = self.special_log1pmx(&first);
        let second = self.special_log1pmx(&second);
        let first = self.mul(a, &first);
        let second = self.mul(b, &second);
        let loga = self.log(a);
        let logb = self.log(b);
        let logs = self.log(&sum);
        let scale = self.add(&loga, &logb);
        let scale = self.sub(&scale, &logs);
        let scale = self.mul(&scale, &half);
        let constant = self.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
        let scale = self.sub(&scale, &constant);
        let ra = self.stirling_remainder(a);
        let rb = self.stirling_remainder(b);
        let rs = self.stirling_remainder(&sum);
        let scale = self.sub(&scale, &ra);
        let scale = self.sub(&scale, &rb);
        let scale = self.add(&scale, &rs);
        let value = self.add(&first, &second);
        let value = self.add(&value, &scale);
        self.select(&stable, &value, &direct)
    }

    fn ratio_tolerance(&self) -> f64 {
        match self.dtype {
            Dtype::F32 => 4.0 * f32::EPSILON as f64,
            Dtype::F64 => 4.0 * f64::EPSILON,
        }
    }

    fn lentz_nonzero(&mut self, x: &Value) -> Value {
        let tiny = self.scalar(match self.dtype {
            Dtype::F32 => 1e-30,
            Dtype::F64 => 1e-300,
        });
        let magnitude = self.abs(x);
        let small = self.compare("LT", &magnitude, &tiny);
        let zero = self.scalar(0.0);
        let negative = self.compare("LT", x, &zero);
        let minus = self.neg(&tiny);
        let signed = self.select(&negative, &minus, &tiny);
        self.select(&small, &signed, x)
    }

    pub(crate) fn regularized_gamma(&mut self, a: &Value, x: &Value) -> Value {
        self.regularized_gamma_tail(a, x, false, false, false)
    }

    pub(super) fn regularized_gamma_tail(
        &mut self,
        a: &Value,
        x: &Value,
        upper: bool,
        logarithmic: bool,
        checked: bool,
    ) -> Value {
        let a = self.convert(a, ElemKind::Real);
        let x = self.convert(x, ElemKind::Real);
        let (a, x) = self.broadcast_pair(&a, &x);
        let zero = self.constant_like(0.0, &x);
        let one = self.constant_like(1.0, &x);
        let positive_a = self.compare("GT", &a, &zero);
        let nonnegative_x = self.compare("GE", &x, &zero);
        let valid = self.and(&positive_a, &nonnegative_x);
        let finite_a = self.special_finite(&a);
        let finite_x = self.special_finite(&x);
        let valid = self.and(&valid, &finite_a);
        if checked {
            self.record_runtime_error(&valid, 12);
        }
        let a = self.select(&valid, &a, &one);
        let finite_input = self.and(&valid, &finite_x);
        let infinity = self.constant_like(f64::INFINITY, &x);
        let infinite_x = self.compare("EQ", &x, &infinity);
        let x = self.select(&finite_input, &x, &one);
        let actual_a = a;
        let actual_x = x;
        let temme_threshold = self.scalar(1e6);
        let large_shape = self.compare("GE", &actual_a, &temme_threshold);
        let difference = self.sub(&actual_x, &actual_a);
        let delta = self.div(&difference, &actual_a);
        let magnitude = self.abs(&delta);
        let half = self.scalar(0.5);
        let near = self.compare("LT", &magnitude, &half);
        let use_temme = self.and(&large_shape, &near);
        let small_shape = self.compare("LT", &actual_a, &half);
        let small_argument = self.compare("LE", &actual_x, &one);
        let positive_argument = self.compare("GT", &actual_x, &zero);
        let use_small = self.and(&small_shape, &small_argument);
        let use_small = self.and(&use_small, &positive_argument);
        let special = self.or(&use_temme, &use_small);
        let a = self.select(&special, &one, &actual_a);
        let x = self.select(&special, &one, &actual_x);
        let threshold = self.add(&a, &one);
        let use_series = self.compare("LT", &x, &threshold);
        let xs = self.min(&x, &threshold);
        let xs_positive = self.compare("GT", &xs, &zero);
        let safe_xs = self.select(&xs_positive, &xs, &one);
        let initial = self.div(&one, &a);
        let tolerance = self.scalar(self.ratio_tolerance());
        let root_a = self.sqrt(&a);
        let twelve = self.scalar(12.0);
        let sixty_four = self.scalar(64.0);
        let series_budget = self.mul(&root_a, &twelve);
        let series_budget = self.add(&series_budget, &sixty_four);
        let maximum = self.scalar(16384.0);
        let series_budget = self.min(&series_budget, &maximum);
        let series = self.special_loop(
            16384,
            &series_budget,
            &[initial.clone(), initial, zero.clone()],
            |e, index, state| {
                let n = e.convert(index, ElemKind::Real);
                let n = e.add(&n, &one);
                let denominator = e.add(&a, &n);
                let factor = e.div(&xs, &denominator);
                let term = e.mul(&state[0], &factor);
                let sum = e.add(&state[1], &term);
                let mag_term = e.abs(&term);
                let mag_sum = e.abs(&sum);
                let eps_sum = e.mul(&tolerance, &mag_sum);
                let converged = e.compare("LE", &mag_term, &eps_sum);
                let done = e.compare("NE", &state[2], &zero);
                let keep = e.or(&done, &converged);
                let sum = e.select(&done, &state[1], &sum);
                let term = e.select(&keep, &zero, &term);
                let flag = e.convert(&keep, ElemKind::Real);
                vec![term, sum, flag]
            },
        );
        let log_scale = self.gamma_log_prefactor(&a, &safe_xs);
        let log_sum = self.log(&series[1]);
        let log_p = self.add(&log_scale, &log_sum);
        let minus_infinity = self.scalar(f64::NEG_INFINITY);
        let log_p = self.select(&xs_positive, &log_p, &minus_infinity);

        let xc = self.max(&x, &threshold);
        let b = self.add(&xc, &one);
        let b = self.sub(&b, &a);
        let inverse = self.div(&one, &b);
        let c = self.scalar(match self.dtype {
            Dtype::F32 => 1e30,
            Dtype::F64 => 1e300,
        });
        let c = self.add(&zero, &c);
        let fourth_root = self.sqrt(&root_a);
        let sixteen = self.scalar(16.0);
        let fraction_budget = self.mul(&fourth_root, &sixteen);
        let fraction_budget = self.add(&fraction_budget, &sixty_four);
        let maximum = self.scalar(4096.0);
        let fraction_budget = self.min(&fraction_budget, &maximum);
        let fraction = self.special_loop(
            4096,
            &fraction_budget,
            &[b, c, inverse.clone(), inverse, zero.clone()],
            |e, index, state| {
                let n = e.convert(index, ElemKind::Real);
                let n = e.add(&n, &one);
                let difference = e.sub(&a, &n);
                let coefficient = e.mul(&n, &difference);
                let two = e.scalar(2.0);
                let b = e.add(&state[0], &two);
                let product = e.mul(&coefficient, &state[2]);
                let d = e.add(&b, &product);
                let d = e.lentz_nonzero(&d);
                let d = e.div(&one, &d);
                let ratio = e.div(&coefficient, &state[1]);
                let c = e.add(&b, &ratio);
                let c = e.lentz_nonzero(&c);
                let delta = e.mul(&c, &d);
                let h = e.mul(&state[3], &delta);
                let diff = e.sub(&delta, &one);
                let diff = e.abs(&diff);
                let converged = e.compare("LE", &diff, &tolerance);
                let done = e.compare("NE", &state[4], &zero);
                let keep = e.or(&done, &converged);
                let h = e.select(&done, &state[3], &h);
                let flag = e.convert(&keep, ElemKind::Real);
                vec![b, c, d, h, flag]
            },
        );
        let log_scale = self.gamma_log_prefactor(&a, &xc);
        let log_fraction = self.log(&fraction[3]);
        let log_q = self.add(&log_scale, &log_fraction);
        let (lower, higher) = if logarithmic {
            let other_p = self.special_log1mexp(&log_q);
            let other_q = self.special_log1mexp(&log_p);
            (
                self.select(&use_series, &log_p, &other_p),
                self.select(&use_series, &other_q, &log_q),
            )
        } else {
            let p = self.exp(&log_p);
            let q = self.exp(&log_q);
            let other_p = self.expm1(&log_q);
            let other_p = self.neg(&other_p);
            let other_q = self.expm1(&log_p);
            let other_q = self.neg(&other_q);
            (
                self.select(&use_series, &p, &other_p),
                self.select(&use_series, &other_q, &q),
            )
        };
        let value = if upper { higher } else { lower };
        let value = if !logarithmic {
            let at_zero = self.compare("EQ", &actual_x, &zero);
            let first_shape = self.compare("EQ", &actual_a, &one);
            let endpoint = self.select(&first_shape, &actual_x, &zero);
            let endpoint = if upper {
                self.sub(&one, &endpoint)
            } else {
                endpoint
            };
            self.select(&at_zero, &endpoint, &value)
        } else {
            value
        };
        let temme_a = self.select(&use_temme, &actual_a, &temme_threshold);
        let temme_x = self.select(&use_temme, &actual_x, &temme_threshold);
        let temme = self.temme_gamma_log(&temme_a, &temme_x, upper);
        let temme = if logarithmic { temme } else { self.exp(&temme) };
        let value = self.select(&use_temme, &temme, &value);
        let small_a = self.scalar(0.1);
        let small_a = self.select(&use_small, &actual_a, &small_a);
        let small_x = self.select(&use_small, &actual_x, &half);
        let (small_p, small_q) = self.small_gamma_logs(&small_a, &small_x);
        let small_value = if upper { small_q } else { small_p };
        let small_value = if logarithmic {
            small_value
        } else {
            self.exp(&small_value)
        };
        let value = self.select(&use_small, &small_value, &value);
        let infinite_endpoint = if logarithmic {
            if upper {
                self.scalar(f64::NEG_INFINITY)
            } else {
                zero.clone()
            }
        } else if upper {
            zero.clone()
        } else {
            one.clone()
        };
        let value = self.select(&infinite_x, &infinite_endpoint, &value);
        let converged = self.select(&use_series, &series[2], &fraction[4]);
        let converged = self.compare("NE", &converged, &zero);
        let converged = self.or(&converged, &use_temme);
        let converged = self.or(&converged, &use_small);
        let invalid = self.not(&valid);
        let status_ok = self.or(&invalid, &converged);
        if checked {
            self.record_runtime_error(&status_ok, 10);
        }
        let valid = self.and(&valid, &converged);
        let nan = self.scalar(f64::NAN);
        self.select(&valid, &value, &nan)
    }

    pub(super) fn regularized_beta(&mut self, a: &Value, b: &Value, x: &Value) -> Value {
        self.regularized_beta_tail(a, b, x, false, false, false)
    }

    pub(super) fn regularized_beta_tail(
        &mut self,
        a: &Value,
        b: &Value,
        x: &Value,
        upper: bool,
        logarithmic: bool,
        checked: bool,
    ) -> Value {
        let a = self.convert(a, ElemKind::Real);
        let b = self.convert(b, ElemKind::Real);
        let x = self.convert(x, ElemKind::Real);
        let (a, x) = self.broadcast_pair(&a, &x);
        let (b, x) = self.broadcast_pair(&b, &x);
        let zero = self.constant_like(0.0, &x);
        let one = self.constant_like(1.0, &x);
        let a_positive = self.compare("GT", &a, &zero);
        let b_positive = self.compare("GT", &b, &zero);
        let x_lower = self.compare("GE", &x, &zero);
        let x_upper = self.compare("LE", &x, &one);
        let valid_shapes = self.and(&a_positive, &b_positive);
        let valid_x = self.and(&x_lower, &x_upper);
        let valid = self.and(&valid_shapes, &valid_x);
        let finite_a = self.special_finite(&a);
        let finite_b = self.special_finite(&b);
        let finite_x = self.special_finite(&x);
        let finite = self.and(&finite_a, &finite_b);
        let finite = self.and(&finite, &finite_x);
        let valid = self.and(&valid, &finite);
        if checked {
            self.record_runtime_error(&valid, 12);
        }
        let a = self.select(&valid, &a, &one);
        let b = self.select(&valid, &b, &one);
        let half = self.scalar(0.5);
        let x = self.select(&valid, &x, &half);
        let two = self.scalar(2.0);
        let numerator = self.add(&a, &one);
        let sum = self.add(&a, &b);
        let denominator = self.add(&sum, &two);
        let threshold = self.div(&numerator, &denominator);
        let flip = self.compare("GT", &x, &threshold);
        let left = self.select(&flip, &b, &a);
        let right = self.select(&flip, &a, &b);
        let complement = self.sub(&one, &x);
        let z = self.select(&flip, &complement, &x);
        let interior = self.compare("GT", &z, &zero);
        let half = self.scalar(0.5);
        let z = self.select(&interior, &z, &half);
        let qab = self.add(&left, &right);
        let qap = self.add(&left, &one);
        let qam = self.sub(&left, &one);
        let initial = self.mul(&qab, &z);
        let initial = self.div(&initial, &qap);
        let initial = self.sub(&one, &initial);
        let initial = self.lentz_nonzero(&initial);
        let initial = self.div(&one, &initial);
        let tolerance = self.scalar(self.ratio_tolerance());
        let root = self.sqrt(&qab);
        let fourth_root = self.sqrt(&root);
        let sixteen = self.scalar(16.0);
        let sixty_four = self.scalar(64.0);
        let budget = self.mul(&fourth_root, &sixteen);
        let budget = self.add(&budget, &sixty_four);
        let state = self.special_loop(
            4096,
            &budget,
            &[one.clone(), initial.clone(), initial, zero.clone()],
            |e, index, state| {
                let m = e.convert(index, ElemKind::Real);
                let m = e.add(&m, &one);
                let m2 = e.mul(&m, &two);
                let bm = e.sub(&right, &m);
                let numerator = e.mul(&m, &bm);
                let numerator = e.mul(&numerator, &z);
                let d1 = e.add(&qam, &m2);
                let d2 = e.add(&left, &m2);
                let denominator = e.mul(&d1, &d2);
                let aa = e.div(&numerator, &denominator);
                let (c, d, h) = e.beta_lentz_step(&aa, &state[0], &state[1], &state[2]);
                let am = e.add(&left, &m);
                let abm = e.add(&qab, &m);
                let numerator = e.mul(&am, &abm);
                let numerator = e.mul(&numerator, &z);
                let numerator = e.neg(&numerator);
                let d1 = e.add(&left, &m2);
                let d2 = e.add(&qap, &m2);
                let denominator = e.mul(&d1, &d2);
                let aa = e.div(&numerator, &denominator);
                let (c, d, h) = e.beta_lentz_step(&aa, &c, &d, &h);
                let delta = e.mul(&c, &d);
                let difference = e.sub(&delta, &one);
                let difference = e.abs(&difference);
                let converged = e.compare("LE", &difference, &tolerance);
                let done = e.compare("NE", &state[3], &zero);
                let keep = e.or(&done, &converged);
                let h = e.select(&done, &state[2], &h);
                let flag = e.convert(&keep, ElemKind::Real);
                vec![c, d, h, flag]
            },
        );
        let log_scale = self.beta_log_prefactor(&left, &right, &z);
        let log_fraction = self.log(&state[2]);
        let log_left = self.log(&left);
        let log_fraction = self.sub(&log_fraction, &log_left);
        let log_value = self.add(&log_scale, &log_fraction);
        let small_threshold = self.scalar(0.01);
        let small_shape = self.compare("LT", &left, &small_threshold);
        let product = self.mul(&z, &right);
        let small_product = self.compare("LE", &product, &one);
        let use_small = self.and(&small_shape, &small_product);
        let small_a = self.scalar(0.001);
        let small_a = self.select(&use_small, &left, &small_a);
        let small_b = self.scalar(2.0);
        let small_b = self.select(&use_small, &right, &small_b);
        let small_x = self.scalar(0.25);
        let small_x = self.select(&use_small, &z, &small_x);
        let (small_log_p, small_log_q, small_done) =
            self.small_beta_logs(&small_a, &small_b, &small_x);
        let log_value = self.select(&use_small, &small_log_p, &log_value);
        let (tail, complement) = if logarithmic {
            let direct_complement = self.special_log1mexp(&log_value);
            let complement = self.select(&use_small, &small_log_q, &direct_complement);
            (log_value.clone(), complement)
        } else {
            let tail = self.exp(&log_value);
            let complement = self.expm1(&log_value);
            let complement = self.neg(&complement);
            let small_complement = self.exp(&small_log_q);
            let complement = self.select(&use_small, &small_complement, &complement);
            (tail, complement)
        };
        let flipped = if upper { self.not(&flip) } else { flip.clone() };
        let value = self.select(&flipped, &complement, &tail);
        let endpoint = self.select(&flip, &one, &zero);
        let first_left = self.compare("EQ", &a, &one);
        let unflipped = self.not(&flip);
        let first_left = self.and(&first_left, &unflipped);
        let left_endpoint = self.mul(&b, &x);
        let endpoint = self.select(&first_left, &left_endpoint, &endpoint);
        let first_right = self.compare("EQ", &b, &one);
        let first_right = self.and(&first_right, &flip);
        let complement = self.sub(&one, &x);
        let right_endpoint = self.mul(&a, &complement);
        let right_endpoint = self.sub(&one, &right_endpoint);
        let endpoint = self.select(&first_right, &right_endpoint, &endpoint);
        let endpoint = if upper {
            self.sub(&one, &endpoint)
        } else {
            endpoint
        };
        let endpoint = if logarithmic {
            let minus_infinity = self.scalar(f64::NEG_INFINITY);
            self.select(&flipped, &zero, &minus_infinity)
        } else {
            endpoint
        };
        let value = self.select(&interior, &value, &endpoint);
        let converged = self.select(&use_small, &small_done, &state[3]);
        let converged = self.compare("NE", &converged, &zero);
        let endpoint = self.not(&interior);
        let converged = self.or(&converged, &endpoint);
        let invalid = self.not(&valid);
        let status_ok = self.or(&invalid, &converged);
        if checked {
            self.record_runtime_error(&status_ok, 10);
        }
        let converged = self.and(&valid, &converged);
        let nan = self.scalar(f64::NAN);
        self.select(&converged, &value, &nan)
    }

    fn beta_lentz_step(
        &mut self,
        aa: &Value,
        c: &Value,
        d: &Value,
        h: &Value,
    ) -> (Value, Value, Value) {
        let one = self.scalar(1.0);
        let product = self.mul(aa, d);
        let d = self.add(&one, &product);
        let d = self.lentz_nonzero(&d);
        let d = self.div(&one, &d);
        let ratio = self.div(aa, c);
        let c = self.add(&one, &ratio);
        let c = self.lentz_nonzero(&c);
        let delta = self.mul(&c, &d);
        let h = self.mul(h, &delta);
        (c, d, h)
    }
}
