//! Log density from piecewise Chebyshev expansions of §09's contour integral.
//! Center coefficients use the saddle contour / negative-axis branch cut;
//! the tails retain their analytic exponential and x⁻² factors. This keeps
//! event arrays pointwise, with no per-event quadrature workspace.

use super::*;

use flatppl_core::landau::{C0, REGIONS};

fn chebyshev(e: &mut Emitter, x: &Value, coefficients: &[f64]) -> Value {
    let mut b1 = e.scalar(0.0);
    let mut b2 = e.scalar(0.0);
    let two = e.scalar(2.0);
    let twice_x = e.mul(&two, x);
    for coefficient in coefficients[1..].iter().rev() {
        let product = e.mul(&twice_x, &b1);
        let difference = e.sub(&product, &b2);
        let c = e.scalar(*coefficient);
        let b0 = e.add(&difference, &c);
        b2 = b1;
        b1 = b0;
    }
    let product = e.mul(x, &b1);
    let value = e.sub(&product, &b2);
    let c = e.scalar(coefficients[0]);
    e.add(&value, &c)
}

pub(super) fn logpdf(e: &mut Emitter, p: &Params, x: &Value) -> Result<Value, EmitError> {
    let loc = p.get(e, "loc")?;
    let scale = p.get(e, "scale")?;
    let difference = e.sub(x, &loc);
    let z = e.div(&difference, &scale);
    let magnitude = e.abs(&z);
    let infinity = e.inf_like(&z);
    let finite = e.compare("LT", &magnitude, &infinity);
    let zero = e.scalar(0.0);
    let safe_z = e.select(&finite, &z, &zero);
    let minus_five = e.scalar(-5.0);
    let left = e.compare("LE", &safe_z, &minus_five);
    let left_x = e.select(&left, &safe_z, &minus_five);
    let one = e.scalar(1.0);
    let exponent = e.add(&left_x, &one);
    let u = e.exp(&exponent);
    let negative = e.neg(&exponent);
    let c = e.exp(&negative);
    let coefficient = e.scalar(2.0 / (-4.0_f64).exp());
    let coordinate = e.mul(&coefficient, &u);
    let coordinate = e.sub(&coordinate, &one);
    let correction = chebyshev(e, &coordinate, C0);
    let half = e.scalar(0.5);
    let half_exponent = e.mul(&half, &exponent);
    let left_base = e.neg(&c);
    let left_base = e.sub(&left_base, &half_exponent);
    let constant = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
    let left_base = e.sub(&left_base, &constant);
    let mut result = e.add(&left_base, &correction);
    let twenty = e.scalar(20.0);
    let right = e.compare("GE", &safe_z, &twenty);
    let right_x = e.select(&right, &safe_z, &twenty);
    let log_right = e.log(&right_x);
    let two = e.scalar(2.0);
    let twice_log_right = e.mul(&two, &log_right);
    for region in REGIONS {
        let raw = if region.logarithmic {
            &log_right
        } else {
            &safe_z
        };
        let lower = e.scalar(region.lower);
        let upper = e.scalar(region.upper);
        let above = e.compare("GE", raw, &lower);
        let below = e.compare("LE", raw, &upper);
        let within = e.and(&above, &below);
        let within = if region.logarithmic {
            e.and(&within, &right)
        } else {
            within
        };
        let midpoint = e.scalar(0.5 * (region.lower + region.upper));
        let input = e.select(&within, raw, &midpoint);
        let distance = e.sub(&input, &midpoint);
        let multiplier = e.scalar(2.0 / (region.upper - region.lower));
        let coordinate = e.mul(&multiplier, &distance);
        let value = chebyshev(e, &coordinate, region.coefficients);
        let value = if region.logarithmic {
            e.sub(&value, &twice_log_right)
        } else {
            value
        };
        result = e.select(&within, &value, &result);
    }
    // Watson expansion of the branch-cut integral, through x⁻⁴. The first
    // omitted relative term is O((log(x)/x)³), negligible from 10^8 upward.
    let threshold = e.scalar(1e8);
    let far = e.compare("GE", &safe_z, &threshold);
    let far_x = e.select(&far, &safe_z, &threshold);
    let log_x = e.log(&far_x);
    let inverse = e.div(&one, &far_x);
    let twice_log = e.mul(&two, &log_x);
    let euler = std::f64::consts::EULER_GAMMA;
    let constant = e.scalar(2.0 * euler - 3.0);
    let first = e.add(&twice_log, &constant);
    let constant = e.scalar(euler - 11.0 / 6.0);
    let shifted = e.add(&log_x, &constant);
    let square = e.mul(&shifted, &shifted);
    let three = e.scalar(3.0);
    let second = e.mul(&three, &square);
    let constant = e.scalar(std::f64::consts::PI.powi(2) / 2.0 + 49.0 / 12.0);
    let second = e.sub(&second, &constant);
    let second = e.mul(&second, &inverse);
    let relative = e.add(&first, &second);
    let relative = e.mul(&relative, &inverse);
    let correction = e.log1p(&relative);
    let far_value = e.sub(&correction, &twice_log);
    result = e.select(&far, &far_value, &result);
    let negative_infinity = e.constant_like(f64::NEG_INFINITY, &result);
    result = e.select(&finite, &result, &negative_infinity);
    let nan = e.compare("NE", &z, &z);
    let nan_value = e.constant_like(f64::NAN, &result);
    result = e.select(&nan, &nan_value, &result);
    let log_scale = e.log(&scale);
    Ok(e.sub(&result, &log_scale))
}
