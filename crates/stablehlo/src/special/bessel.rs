//! Real-order Bessel and Airy functions, DLMF 10.2.2, 10.9.6–7, 10.25.2,
//! 10.32.9, 10.17 and 10.40. Integral rules use composite Gauss–Legendre.
//! Miller's recurrence avoids cancellation in J when the order exceeds x/2.

use super::*;

impl Emitter<'_> {
    pub(super) fn bessel_principal_parts(
        &mut self,
        member: &str,
        order: &Value,
        x: &Value,
    ) -> (Value, Value) {
        let magnitude = self.bessel_real(member, order, x);
        let (sine, cosine) = self.special_sincospi(order);
        let zero = self.scalar(0.0);
        let negative = self.compare("LT", x, &zero);
        let real_part = self.mul(&cosine, &magnitude);
        let imag_part = self.mul(&sine, &magnitude);
        (
            self.select(&negative, &real_part, &magnitude),
            self.select(&negative, &imag_part, &zero),
        )
    }
    fn special_quad(
        &mut self,
        upper: &Value,
        template: &Value,
        frequency: &Value,
        mut f: impl FnMut(&mut Self, &Value) -> Value,
    ) -> Value {
        // Resolve the oscillation before testing global
        // refinement. The roundoff floor scales with the integral of |f|.
        let four = self.scalar(4.0);
        let minimum = self.scalar(32.0);
        let requested = self.div(frequency, &four);
        let requested = self.ceil(&requested);
        let requested = self.max(&requested, &minimum);
        let rank = shape(&requested.ty).len();
        let requested = self.axes_view(
            &requested,
            Axes {
                batch: 0,
                layers: if rank == 0 { vec![] } else { vec![rank] },
            },
        );
        let panels = self.reduce_max(&requested);
        let (coarse, _) = self.gauss_legendre16(upper, template, &panels, 65536, &mut f);
        let two = self.scalar(2.0);
        let panels = self.mul(&panels, &two);
        let (fine, magnitude) = self.gauss_legendre16(upper, template, &panels, 65536, &mut f);
        let eps = self.scalar(match self.dtype {
            Dtype::F32 => 32.0 * f32::EPSILON as f64,
            Dtype::F64 => 32.0 * f64::EPSILON,
        });
        let mut previous = coarse;
        let mut fine = fine;
        let mut magnitude = magnitude;
        let mut panels = panels;
        let mut chosen = self.scalar(f64::NAN);
        let mut accepted = self.bool_value_const(false);
        // Three known trip-count refinements preserve reverse AD in Enzyme.
        // Each failed certificate remains a failure, never an unchecked value.
        for refinement in 0..3 {
            let difference = self.sub(&fine, &previous);
            let difference = self.abs(&difference);
            let tolerance = self.mul(&eps, &magnitude);
            let passed = self.compare("LE", &difference, &tolerance);
            let not_accepted = self.not(&accepted);
            let take = self.and(&not_accepted, &passed);
            chosen = self.select(&take, &fine, &chosen);
            accepted = self.or(&accepted, &passed);
            if refinement < 2 {
                previous = fine;
                panels = self.mul(&panels, &two);
                (fine, magnitude) = self.gauss_legendre16(upper, template, &panels, 65536, &mut f);
            }
        }
        chosen
    }

    fn special_sincospi(&mut self, x: &Value) -> (Value, Value) {
        let half = self.scalar(0.5);
        let shifted = self.add(x, &half);
        let n = self.floor(&shifted);
        let fraction = self.sub(x, &n);
        let half = self.scalar(0.5);
        let two = self.scalar(2.0);
        let one = self.scalar(1.0);
        let half_n = self.mul(&n, &half);
        let half_n = self.floor(&half_n);
        let even = self.mul(&half_n, &two);
        let parity = self.sub(&n, &even);
        let parity = self.mul(&parity, &two);
        let sign = self.sub(&one, &parity);
        let pi = self.scalar(std::f64::consts::PI);
        let angle = self.mul(&fraction, &pi);
        let sine = self.sin(&angle);
        let sine = self.mul(&sine, &sign);
        let abs_fraction = self.abs(&fraction);
        let complement = self.sub(&half, &abs_fraction);
        let complement = self.mul(&complement, &pi);
        let cosine = self.sin(&complement);
        let cosine = self.mul(&cosine, &sign);
        (sine, cosine)
    }

    fn bessel_series(&mut self, order: &Value, x: &Value, modified: bool) -> Value {
        let one = self.constant_like(1.0, x);
        let zero = self.constant_like(0.0, x);
        let half = self.scalar(0.5);
        let half_x = self.mul(x, &half);
        let square = self.mul(&half_x, &half_x);
        let square = if modified { square } else { self.neg(&square) };
        let tolerance = self.scalar(match self.dtype {
            Dtype::F32 => f32::EPSILON as f64,
            Dtype::F64 => f64::EPSILON,
        });
        let scale = self.scalar(match self.dtype {
            Dtype::F32 => 1e15,
            Dtype::F64 => 1e150,
        });
        let inverse_scale = self.div(&one, &scale);
        let log_scale = self.log(&scale);
        let half_order = self.mul(order, &half);
        let order_square = self.mul(&half_order, &half_order);
        let positive_square = self.mul(&half_x, &half_x);
        let radius = self.add(&order_square, &positive_square);
        let radius = self.sqrt(&radius);
        let denominator = self.add(&radius, &half_order);
        let peak = self.div(&positive_square, &denominator);
        let peak_plus_one = self.add(&peak, &one);
        let spread = self.sqrt(&peak_plus_one);
        let twelve = self.scalar(12.0);
        let spread = self.mul(&spread, &twelve);
        let sixty_four = self.scalar(64.0);
        let budget = self.add(&peak, &spread);
        let budget = self.add(&budget, &sixty_four);
        let state = self.special_loop(
            4096,
            &budget,
            &[one.clone(), one.clone(), zero.clone(), zero.clone()],
            |e, index, state| {
                let k = e.convert(index, ElemKind::Real);
                let k = e.add(&k, &one);
                let denominator = e.add(order, &k);
                let denominator = e.mul(&denominator, &k);
                let factor = e.div(&square, &denominator);
                let term = e.mul(&state[0], &factor);
                let sum = e.add(&state[1], &term);
                let mag_term = e.abs(&term);
                let mag_sum = e.abs(&sum);
                let bound = e.mul(&tolerance, &mag_sum);
                let converged = e.compare("LE", &mag_term, &bound);
                let done = e.compare("NE", &state[3], &zero);
                let keep = e.or(&done, &converged);
                let sum = e.select(&done, &state[1], &sum);
                let large = e.compare("GT", &mag_sum, &scale);
                let factor = e.select(&large, &inverse_scale, &one);
                let sum = e.mul(&sum, &factor);
                let term = e.mul(&term, &factor);
                let term = e.select(&keep, &zero, &term);
                let add_scale = e.select(&large, &log_scale, &zero);
                let saved_scale = e.add(&state[2], &add_scale);
                let flag = e.convert(&keep, ElemKind::Real);
                vec![term, sum, saved_scale, flag]
            },
        );
        let logx = self.log(&half_x);
        let power = self.mul(order, &logx);
        let op1 = self.add(order, &one);
        let gamma = self.lgamma(&op1);
        let log_value = self.sub(&power, &gamma);
        let magnitude = self.abs(&state[1]);
        let log_sum = self.log(&magnitude);
        let log_value = self.add(&log_value, &log_sum);
        let log_value = self.add(&log_value, &state[2]);
        let value = self.exp(&log_value);
        let negative = self.compare("LT", &state[1], &zero);
        let minus = self.neg(&value);
        let value = self.select(&negative, &minus, &value);
        let converged = self.compare("NE", &state[3], &zero);
        let nan = self.scalar(f64::NAN);
        self.select(&converged, &value, &nan)
    }

    fn bessel_hankel(&mut self, order: &Value, x: &Value) -> (Value, Value) {
        let one = self.constant_like(1.0, x);
        let zero = self.constant_like(0.0, x);
        let four = self.scalar(4.0);
        let square = self.mul(order, order);
        let mu = self.mul(&square, &four);
        let initial = [one.clone(), one, zero.clone()];
        let axes = self.axes_of(&initial[0]);
        let state = self.matrix_loop(32, 0, &initial, |e, index, state| {
            let state: Vec<_> = state.iter().map(|v| e.axes_view(v, axes.clone())).collect();
            let k = e.convert(index, ElemKind::Real);
            let one = e.scalar(1.0);
            let two = e.scalar(2.0);
            let four = e.scalar(4.0);
            let eight = e.scalar(8.0);
            let k = e.add(&k, &one);
            let odd = e.mul(&k, &two);
            let odd = e.sub(&odd, &one);
            let odd = e.mul(&odd, &odd);
            let numerator = e.sub(&mu, &odd);
            let denominator = e.mul(&eight, &k);
            let denominator = e.mul(&denominator, x);
            let factor = e.div(&numerator, &denominator);
            let term = e.mul(&state[0], &factor);
            let half_k = e.div(&k, &two);
            let floor_half = e.floor(&half_k);
            let twice = e.mul(&floor_half, &two);
            let rem2 = e.sub(&k, &twice);
            let is_even = e.compare("EQ", &rem2, &zero);
            let quarter = e.div(&k, &four);
            let quarter = e.floor(&quarter);
            let multiples = e.mul(&quarter, &four);
            let rem4 = e.sub(&k, &multiples);
            let positive = e.compare("LE", &rem4, &one);
            let minus = e.neg(&term);
            let signed = e.select(&positive, &term, &minus);
            let p_term = e.select(&is_even, &signed, &zero);
            let q_term = e.select(&is_even, &zero, &signed);
            let p = e.add(&state[1], &p_term);
            let q = e.add(&state[2], &q_term);
            vec![term, p, q]
        });
        let state: Vec<_> = state
            .iter()
            .map(|v| self.axes_view(v, axes.clone()))
            .collect();
        let pi = self.scalar(std::f64::consts::PI);
        let half_pi = self.scalar(std::f64::consts::FRAC_PI_2);
        let quarter_pi = self.scalar(std::f64::consts::FRAC_PI_4);
        let shift = self.mul(order, &half_pi);
        let phase = self.sub(x, &shift);
        let phase = self.sub(&phase, &quarter_pi);
        let cosine = self.cos(&phase);
        let sine = self.sin(&phase);
        let pc = self.mul(&state[1], &cosine);
        let qs = self.mul(&state[2], &sine);
        let ps = self.mul(&state[1], &sine);
        let qc = self.mul(&state[2], &cosine);
        let j = self.sub(&pc, &qs);
        let y = self.add(&ps, &qc);
        let two = self.scalar(2.0);
        let denominator = self.mul(&pi, x);
        let scale = self.div(&two, &denominator);
        let scale = self.sqrt(&scale);
        (self.mul(&scale, &j), self.mul(&scale, &y))
    }

    fn bessel_tail(&mut self, order: &Value, x: &Value, kind: u8) -> Value {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let two = self.scalar(2.0);
        let fifty = self.scalar(50.0);
        let ratio = self.div(order, x);
        let saddle = if kind == 2 {
            self.add(&zero, &zero)
        } else {
            self.asinh(&ratio)
        };
        let at_zero = self.bessel_tail_exponent(order, x, &zero, kind);
        let at_saddle = self.bessel_tail_exponent(order, x, &saddle, kind);
        let peak = self.max(&at_zero, &at_saddle);
        let upper = self.add(order, &fifty);
        let upper = self.div(&upper, x);
        let upper = self.log1p(&upper);
        let upper = self.add(&upper, &two);
        let upper = self.max(&upper, &one);
        let frequency = self.scalar(0.0);
        let value = self.special_quad(&upper, x, &frequency, |e, t| {
            let exponent = e.bessel_tail_exponent(order, x, t, kind);
            let exponent = e.sub(&exponent, &peak);
            e.exp(&exponent)
        });
        let log_value = self.log(&value);
        let log_value = self.add(&peak, &log_value);
        self.exp(&log_value)
    }

    // kind 0: K, 1: growing Y correction, 2: decaying J/Y correction.
    fn bessel_tail_exponent(&mut self, order: &Value, x: &Value, t: &Value, kind: u8) -> Value {
        let logx = self.log(x);
        let log_two = self.scalar(std::f64::consts::LN_2);
        let base = self.sub(&logx, &log_two);
        let forward = self.add(&base, t);
        let forward = self.exp(&forward);
        let backward = self.sub(&base, t);
        let backward = self.exp(&backward);
        let vt = self.mul(order, t);
        if kind == 0 {
            let decay = self.add(&forward, &backward);
            let decay = self.neg(&decay);
            let magnitude = self.abs(&vt);
            let minus_two = self.scalar(-2.0);
            let correction = self.mul(&magnitude, &minus_two);
            let correction = self.exp(&correction);
            let correction = self.log1p(&correction);
            let logcosh = self.add(&magnitude, &correction);
            let logcosh = self.sub(&logcosh, &log_two);
            self.add(&decay, &logcosh)
        } else {
            let decay = self.sub(&backward, &forward);
            if kind == 1 {
                self.add(&decay, &vt)
            } else {
                self.sub(&decay, &vt)
            }
        }
    }

    fn bessel_j_base(&mut self, order: &Value, x: &Value) -> Value {
        let ten = self.scalar(10.0);
        let one = self.scalar(1.0);
        let eight = self.scalar(8.0);
        let op1 = self.add(order, &one);
        let limit = self.mul(&eight, &op1);
        let limit = self.sqrt(&limit);
        let limit = self.max(&limit, &ten);
        let xs = self.min(x, &limit);
        let series = self.bessel_series(order, &xs, false);
        let use_series = self.compare("LT", x, &limit);
        let pi = self.scalar(std::f64::consts::PI);
        let frequency = self.add(order, x);
        let integral = self.special_quad(&pi, x, &frequency, |e, t| {
            let phase = e.mul(order, t);
            let sin = e.sin(t);
            let phase2 = e.mul(x, &sin);
            let phase = e.sub(&phase, &phase2);
            e.cos(&phase)
        });
        let (sinpi, _) = self.special_sincospi(order);
        let tail = self.bessel_tail(order, x, 2);
        let tail = self.mul(&sinpi, &tail);
        let integral = self.sub(&integral, &tail);
        let integral = self.div(&integral, &pi);
        let two = self.scalar(2.0);
        let forty = self.scalar(40.0);
        let square = self.mul(order, order);
        let cutoff = self.mul(&two, &square);
        let cutoff = self.max(&cutoff, &forty);
        let large = self.compare("GT", x, &cutoff);
        let safe_large = self.max(x, &cutoff);
        let (asymptotic, _) = self.bessel_hankel(order, &safe_large);
        let value = self.select(&large, &asymptotic, &integral);
        self.select(&use_series, &series, &value)
    }

    fn bessel_positive_j(&mut self, order: &Value, x: &Value) -> Value {
        let base = self.bessel_j_base(order, x);
        let integer = self.floor(order);
        let fraction = self.sub(order, &integer);
        let one = self.scalar(1.0);
        let next_fraction = self.add(&fraction, &one);
        let j0 = self.bessel_j_base(&fraction, x);
        let j1 = self.bessel_j_base(&next_fraction, x);
        let top = self.max(order, x);
        let top = self.ceil(&top);
        let forty = self.scalar(40.0);
        let extra = self.mul(&forty, x);
        let extra = self.sqrt(&extra);
        let extra = self.ceil(&extra);
        let thirty_two = self.scalar(32.0);
        let extra = self.add(&extra, &thirty_two);
        let top = self.add(&top, &extra);
        let zero = self.constant_like(0.0, x);
        let initial_one = self.constant_like(1.0, x);
        let scale = self.scalar(match self.dtype {
            Dtype::F32 => 1e20,
            Dtype::F64 => 1e150,
        });
        let inverse_scale = self.div(&one, &scale);
        let state = self.special_loop(
            4096,
            &top,
            &[initial_one, zero.clone(), zero.clone(), zero.clone()],
            |e, index, state| {
                let i = e.convert(index, ElemKind::Real);
                let n = e.sub(&top, &i);
                let active = e.compare("GT", &n, &zero);
                let n = e.max(&n, &zero);
                let selected = e.compare("EQ", &n, &integer);
                let selected = e.select(&selected, &state[0], &state[2]);
                let nf = e.add(&n, &fraction);
                let two = e.scalar(2.0);
                let coefficient = e.mul(&two, &nf);
                let coefficient = e.div(&coefficient, x);
                let current = e.mul(&coefficient, &state[0]);
                let current = e.sub(&current, &state[1]);
                let magnitude = e.abs(&current);
                let large = e.compare("GT", &magnitude, &scale);
                let factor = e.select(&large, &inverse_scale, &one);
                let current = e.mul(&current, &factor);
                let next = e.mul(&state[0], &factor);
                let selected = e.mul(&selected, &factor);
                let current = e.select(&active, &current, &state[0]);
                let next = e.select(&active, &next, &state[1]);
                let selected = e.select(&active, &selected, &state[2]);
                let done = e.compare("LE", &n, &one);
                let done = e.convert(&done, ElemKind::Real);
                vec![current, next, selected, done]
            },
        );
        let a0 = self.abs(&state[0]);
        let a1 = self.abs(&state[1]);
        let normalize_zero = self.compare("GT", &a0, &a1);
        let denominator = self.select(&normalize_zero, &state[0], &state[1]);
        let normalization = self.select(&normalize_zero, &j0, &j1);
        let ratio = self.div(&state[2], &denominator);
        let miller = self.mul(&ratio, &normalization);
        let eight = self.scalar(8.0);
        let large_order = self.compare("GT", order, &eight);
        let half = self.scalar(0.5);
        let half_x = self.mul(x, &half);
        let order_above = self.compare("GT", order, &half_x);
        let use_miller = self.and(&large_order, &order_above);
        let square = self.mul(x, x);
        let limit = self.add(order, &one);
        let limit = self.mul(&limit, &eight);
        let not_series = self.compare("GE", &square, &limit);
        let ten = self.scalar(10.0);
        let not_small = self.compare("GE", x, &ten);
        let not_series = self.and(&not_series, &not_small);
        let use_miller = self.and(&use_miller, &not_series);
        self.select(&use_miller, &miller, &base)
    }

    fn bessel_positive_y(&mut self, order: &Value, x: &Value) -> Value {
        let pi = self.scalar(std::f64::consts::PI);
        let frequency = self.add(order, x);
        let integral = self.special_quad(&pi, x, &frequency, |e, t| {
            let sine = e.sin(t);
            let left = e.mul(x, &sine);
            let right = e.mul(order, t);
            let phase = e.sub(&left, &right);
            e.sin(&phase)
        });
        let (_, cospi) = self.special_sincospi(order);
        let positive = self.bessel_tail(order, x, 1);
        let negative = self.bessel_tail(order, x, 2);
        let correction = self.mul(&cospi, &negative);
        let integral = self.sub(&integral, &positive);
        let integral = self.sub(&integral, &correction);
        let value = self.div(&integral, &pi);
        let two = self.scalar(2.0);
        let forty = self.scalar(40.0);
        let square = self.mul(order, order);
        let cutoff = self.mul(&two, &square);
        let cutoff = self.max(&cutoff, &forty);
        let large = self.compare("GT", x, &cutoff);
        let safe_large = self.max(x, &cutoff);
        let (_, asymptotic) = self.bessel_hankel(order, &safe_large);
        self.select(&large, &asymptotic, &value)
    }

    fn bessel_positive_i(&mut self, order: &Value, x: &Value) -> Value {
        let square = self.mul(order, order);
        let two = self.scalar(2.0);
        let forty = self.scalar(40.0);
        let cutoff = self.mul(&two, &square);
        let cutoff = self.max(&cutoff, &forty);
        let safe_series = self.min(x, &cutoff);
        let series = self.bessel_series(order, &safe_series, true);
        let safe_large = self.max(x, &cutoff);
        let one = self.constant_like(1.0, x);
        let four = self.scalar(4.0);
        let mu = self.mul(&square, &four);
        let axes = self.axes_of(&one);
        let state = self.matrix_loop(32, 0, &[one.clone(), one], |e, index, state| {
            let state: Vec<_> = state.iter().map(|v| e.axes_view(v, axes.clone())).collect();
            let one = e.scalar(1.0);
            let two = e.scalar(2.0);
            let eight = e.scalar(8.0);
            let k = e.convert(index, ElemKind::Real);
            let k = e.add(&k, &one);
            let odd = e.mul(&k, &two);
            let odd = e.sub(&odd, &one);
            let odd = e.mul(&odd, &odd);
            let numerator = e.sub(&odd, &mu);
            let denominator = e.mul(&eight, &k);
            let denominator = e.mul(&denominator, &safe_large);
            let factor = e.div(&numerator, &denominator);
            let term = e.mul(&state[0], &factor);
            let sum = e.add(&state[1], &term);
            vec![term, sum]
        });
        let state: Vec<_> = state
            .iter()
            .map(|v| self.axes_view(v, axes.clone()))
            .collect();
        let two_pi = self.scalar(2.0 * std::f64::consts::PI);
        let denominator = self.mul(&two_pi, &safe_large);
        let log_denominator = self.log(&denominator);
        let half = self.scalar(0.5);
        let log_denominator = self.mul(&half, &log_denominator);
        let log_sum = self.log(&state[1]);
        let log_value = self.sub(&safe_large, &log_denominator);
        let log_value = self.add(&log_value, &log_sum);
        let asymptotic = self.exp(&log_value);
        let large = self.compare("GT", x, &cutoff);
        self.select(&large, &asymptotic, &series)
    }

    pub(super) fn bessel_real(&mut self, member: &str, order: &Value, x: &Value) -> Value {
        let order = self.convert(order, ElemKind::Real);
        let x = self.convert(x, ElemKind::Real);
        let (order, x) = self.broadcast_pair(&order, &x);
        let positive = self.abs(&x);
        let zero = self.constant_like(0.0, &positive);
        let one = self.constant_like(1.0, &positive);
        let negative_order = self.compare("LT", &order, &zero);
        let minus_order = self.neg(&order);
        let n = self.select(&negative_order, &minus_order, &order);
        let is_positive = self.compare("GT", &positive, &zero);
        let safe_x = self.select(&is_positive, &positive, &one);
        let (sinpi, cospi) = self.special_sincospi(&order);
        let reflected_order = self.select(&negative_order, &n, &one);
        let value = match member {
            "bessel_k" => self.bessel_tail(&n, &safe_x, 0),
            "bessel_y" => {
                let base = self.bessel_positive_y(&n, &safe_x);
                let j = self.bessel_positive_j(&n, &safe_x);
                let c = self.mul(&cospi, &base);
                let s = self.mul(&sinpi, &j);
                let reflected = self.sub(&c, &s);
                self.select(&negative_order, &reflected, &base)
            }
            "bessel_i" => {
                let base = self.bessel_positive_i(&n, &safe_x);
                let k = self.bessel_tail(&reflected_order, &safe_x, 0);
                let coefficient = self.scalar(2.0 / std::f64::consts::PI);
                let coefficient = self.mul(&coefficient, &sinpi);
                let correction = self.mul(&coefficient, &k);
                let reflected = self.sub(&base, &correction);
                self.select(&negative_order, &reflected, &base)
            }
            _ => {
                let base = self.bessel_positive_j(&n, &safe_x);
                let y = self.bessel_positive_y(&reflected_order, &safe_x);
                let c = self.mul(&cospi, &base);
                let s = self.mul(&sinpi, &y);
                let reflected = self.add(&c, &s);
                self.select(&negative_order, &reflected, &base)
            }
        };
        let order_zero = self.compare("EQ", &order, &zero);
        let at_zero = if member == "bessel_k" {
            self.constant_like(f64::INFINITY, &positive)
        } else if member == "bessel_y" {
            self.constant_like(f64::NEG_INFINITY, &positive)
        } else {
            let integer = self.compare("EQ", &sinpi, &zero);
            let noninteger = self.not(&integer);
            let diverges = self.and(&negative_order, &noninteger);
            let negative_sine = self.compare("LT", &sinpi, &zero);
            let infinity = self.constant_like(f64::INFINITY, &positive);
            let minus_infinity = self.neg(&infinity);
            let divergent = self.select(&negative_sine, &infinity, &minus_infinity);
            let endpoint = self.select(&order_zero, &one, &zero);
            let half = self.scalar(0.5);
            let linear = self.mul(&half, &x);
            let linear = if member == "bessel_j" {
                self.mul(&linear, &cospi)
            } else {
                linear
            };
            let first_order = self.compare("EQ", &n, &one);
            let first_order = self.and(&integer, &first_order);
            let endpoint = self.select(&first_order, &linear, &endpoint);
            self.select(&diverges, &divergent, &endpoint)
        };
        self.select(&is_positive, &value, &at_zero)
    }

    pub(super) fn airy_ai(&mut self, x: &Value) -> Value {
        let x = self.convert(x, ElemKind::Real);
        let magnitude = self.abs(&x);
        let one = self.scalar(1.0);
        let near_origin = self.compare("LT", &magnitude, &one);
        let magnitude = self.select(&near_origin, &one, &magnitude);
        let one_third = self.scalar(1.0 / 3.0);
        let minus_third = self.scalar(-1.0 / 3.0);
        let root = self.sqrt(&magnitude);
        let z = self.mul(&magnitude, &root);
        let coefficient = self.scalar(2.0 / 3.0);
        let z = self.mul(&z, &coefficient);
        let zero = self.scalar(0.0);
        let nonzero = self.compare("GT", &magnitude, &zero);
        let safe_z = self.select(&nonzero, &z, &one);
        let k = self.bessel_real("bessel_k", &one_third, &safe_z);
        let jplus = self.bessel_real("bessel_j", &one_third, &safe_z);
        let jminus = self.bessel_real("bessel_j", &minus_third, &safe_z);
        let three = self.scalar(3.0);
        let positive_scale = self.div(&magnitude, &three);
        let positive_scale = self.sqrt(&positive_scale);
        let pi = self.scalar(std::f64::consts::PI);
        let positive_scale = self.div(&positive_scale, &pi);
        let positive = self.mul(&positive_scale, &k);
        let negative = self.add(&jplus, &jminus);
        let negative = self.mul(&negative, &root);
        let negative = self.div(&negative, &three);
        let is_positive = self.compare("GT", &x, &zero);
        let value = self.select(&is_positive, &positive, &negative);
        // Ai''(x) = x Ai(x). This entire series retains Ai'(0) and avoids
        // multiplying singular fractional-order Bessels by sqrt(x) near zero.
        let mut coefficients = [0.0; 33];
        coefficients[0] = 0.355_028_053_887_817_2;
        coefficients[1] = -0.258_819_403_792_806_8;
        for n in 3..coefficients.len() {
            coefficients[n] = coefficients[n - 3] / (n * (n - 1)) as f64;
        }
        let mut series = self.scalar(coefficients[32]);
        for &coefficient in coefficients[..32].iter().rev() {
            series = self.mul(&series, &x);
            let coefficient = self.scalar(coefficient);
            series = self.add(&series, &coefficient);
        }
        self.select(&near_origin, &series, &value)
    }
}
