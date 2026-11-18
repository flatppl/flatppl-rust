//! Periodic continuous samplers.

use super::*;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Sample the centered law and rotate it into the canonical period (-π, π].
pub(super) fn von_mises_sample(e: &mut Emitter, params: &Params) -> Result<Value, EmitError> {
    let ty = sample_lane_type(e);
    let mu = sample_param(e, params, "mu", ElemKind::Real, &ty)?;
    let kappa = sample_param(e, params, "kappa", ElemKind::Real, &ty)?;
    let zero = e.constant(0.0, ty.clone());
    let half = e.scalar(0.5);
    let one = e.scalar(1.0);
    let two = e.scalar(2.0);
    let pi = e.scalar(PI);
    let minus_pi = e.scalar(-PI);
    let half_pi = e.scalar(FRAC_PI_2);
    let infinity = e.scalar(f64::INFINITY);

    let positive = e.compare("GT", &kappa, &zero);
    let finite_kappa = e.compare("LT", &kappa, &infinity);
    let magnitude = e.abs(&mu);
    let finite_mu = e.compare("LT", &magnitude, &infinity);
    let valid = e.and(&positive, &finite_kappa);
    let valid = e.and(&valid, &finite_mu);
    e.record_parameter_validity(&valid);

    // For κ < 1/2 a uniform proposal has envelope ratio bounded by e.
    // Otherwise use Best and Fisher (1979), §4, with
    // r = (1 + sqrt(1 + 4κ²))/(2κ). Compute s = κ(r - 1) directly
    // so neither κ² nor subtraction near r = 1 loses the scale.
    let cauchy = e.compare("GE", &kappa, &half);
    let safe_kappa = e.select(&cauchy, &kappa, &half);
    let inverse = e.div(&half, &safe_kappa);
    let inverse_squared = e.mul(&inverse, &inverse);
    let norm_squared = e.add(&one, &inverse_squared);
    let norm = e.sqrt(&norm_squared);
    let denominator = e.add(&one, &norm);
    let q = e.div(&inverse, &denominator);
    let one_plus_q = e.add(&one, &q);
    let s = e.mul(&half, &one_plus_q);
    let half_r_minus_one = e.mul(&s, &inverse);
    let half_r_plus_one = e.add(&one, &half_r_minus_one);
    let numerator_scale = e.sqrt(&half_r_minus_one);
    let denominator_scale = e.sqrt(&half_r_plus_one);
    let minus_two = e.scalar(-2.0);
    let uniform_scale = e.mul(&minus_two, &kappa);

    let done = e.not(&valid);
    let theta = rejection_draw(e, done, zero, |e, _| {
        let u = e.rng("UNIFORM", &ty);
        let accept = e.rng("UNIFORM", &ty);
        let twice_u = e.mul(&two, &u);
        let signed_u = e.sub(&twice_u, &one);
        let half_angle = e.mul(&half_pi, &signed_u);
        let sin = e.sin(&half_angle);
        let cos = e.cos(&half_angle);
        let cos = e.abs(&cos);
        let sin_squared = e.mul(&sin, &sin);
        let cos_squared = e.mul(&cos, &cos);

        // With z = cos(2t) and f = (1 + rz)/(r + z), evaluate
        // c = κ(r - f) = s + s sin²(t) / ((r - 1)/2 + cos²(t)).
        // The log acceptance probability is log(c) + 1 - c.
        let scaled_sin = e.mul(&s, &sin_squared);
        let denominator = e.add(&half_r_minus_one, &cos_squared);
        let distance = e.div(&scaled_sin, &denominator);
        let c = e.add(&s, &distance);
        let log_c = e.log(&c);
        let one_minus_c = e.sub(&one, &c);
        let cauchy_log_ratio = e.add(&log_c, &one_minus_c);

        // atan2 computes the signed angle without rounding cos(θ) to one.
        let y = e.mul(&numerator_scale, &sin);
        let x = e.mul(&denominator_scale, &cos);
        let angle = e.atan2(&y, &x);
        let cauchy_theta = e.mul(&two, &angle);
        let uniform_theta = e.mul(&pi, &signed_u);
        let uniform_log_ratio = e.mul(&uniform_scale, &sin_squared);
        let value = e.select(&cauchy, &cauchy_theta, &uniform_theta);
        let log_ratio = e.select(&cauchy, &cauchy_log_ratio, &uniform_log_ratio);
        let log_u = e.log(&accept);
        let accepted = e.compare("LT", &log_u, &log_ratio);
        (value, accepted)
    });

    // Reduce mu first so that adding theta retains its variation even when
    // the original location is outside the canonical period.
    let sin_mu = e.sin(&mu);
    let cos_mu = e.cos(&mu);
    let reduced = e.atan2(&sin_mu, &cos_mu);
    let above = e.compare("GT", &mu, &minus_pi);
    let below = e.compare("LE", &mu, &pi);
    let canonical = e.and(&above, &below);
    let center = e.select(&canonical, &mu, &reduced);
    let shifted = e.add(&center, &theta);
    let tau = e.scalar(TAU);
    let above = e.compare("GT", &shifted, &pi);
    let below = e.compare("LE", &shifted, &minus_pi);
    let lower = e.sub(&shifted, &tau);
    let upper = e.add(&shifted, &tau);
    let shifted = e.select(&above, &lower, &shifted);
    let shifted = e.select(&below, &upper, &shifted);
    let undefined = e.scalar(f64::NAN);
    Ok(e.select(&valid, &shifted, &undefined))
}
