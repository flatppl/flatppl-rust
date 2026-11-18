//! Native samplers for scalar probability laws conditioned on a restriction.

use super::*;

pub(super) fn sample(
    e: &mut Emitter,
    params: &Params,
    law: RestrictedSample,
) -> Result<Value, EmitError> {
    if matches!(law, RestrictedSample::NormalInterval) {
        return normal(e, params);
    }

    let ty = lane_type(e);
    let zero = e.constant(0.0, ty.clone());
    let one = e.scalar(1.0);
    let infinity = e.scalar(f64::INFINITY);
    let mut gate_valid = e.compare("EQ", &zero, &zero);
    let mut active = gate_valid.clone();

    if matches!(
        law,
        RestrictedSample::HurdlePoisson | RestrictedSample::HurdleNegativeBinomial
    ) {
        let zero_weight = param(e, params, "zero_weight", &ty)?;
        let positive_weight = param(e, params, "positive_weight", &ty)?;
        let zero_valid = e.compare("GE", &zero_weight, &zero);
        let positive_valid = e.compare("GE", &positive_weight, &zero);
        let scale = e.max(&zero_weight, &positive_weight);
        let nonzero = e.compare("GT", &scale, &zero);
        let finite = e.compare("LT", &scale, &infinity);
        gate_valid = e.and(&zero_valid, &positive_valid);
        gate_valid = e.and(&gate_valid, &nonzero);
        gate_valid = e.and(&gate_valid, &finite);

        // Scale before adding so two large finite weights never overflow.
        let scaled_zero = e.div(&zero_weight, &scale);
        let scaled_positive = e.div(&positive_weight, &scale);
        let total = e.add(&scaled_zero, &scaled_positive);
        let probability = e.div(&scaled_positive, &total);
        let draw = e.rng("UNIFORM", &ty);
        active = e.compare("LT", &draw, &probability);
        active = e.and(&active, &gate_valid);
    }

    let (draw, valid) = match law {
        RestrictedSample::PositivePoisson | RestrictedSample::HurdlePoisson => {
            let rate = param(e, params, "rate", &ty)?;
            let positive = e.compare("GT", &rate, &zero);
            let finite = e.compare("LT", &rate, &infinity);
            let valid = e.and(&positive, &finite);
            let active = e.and(&valid, &active);
            let rate = e.select(&active, &rate, &zero);
            (positive_poisson(e, &rate), valid)
        }
        RestrictedSample::PositiveNegativeBinomial | RestrictedSample::HurdleNegativeBinomial => {
            let alpha = param(e, params, "alpha", &ty)?;
            let beta = param(e, params, "beta", &ty)?;
            let alpha_positive = e.compare("GT", &alpha, &zero);
            let beta_positive = e.compare("GT", &beta, &zero);
            let alpha_finite = e.compare("LT", &alpha, &infinity);
            let beta_finite = e.compare("LT", &beta, &infinity);
            let positive = e.and(&alpha_positive, &beta_positive);
            let finite = e.and(&alpha_finite, &beta_finite);
            let valid = e.and(&positive, &finite);
            let active = e.and(&valid, &active);
            let alpha = e.select(&active, &alpha, &one);
            let beta = e.select(&active, &beta, &one);
            (positive_negative_binomial(e, &alpha, &beta, &active), valid)
        }
        RestrictedSample::NormalInterval => unreachable!(),
    };

    let draw = e.select(&active, &draw, &zero);
    // Existing Poisson kernels count in reals. Refuse draws at the integer
    // resolution limit before conversion to the integer ABI.
    let limit = e.scalar(match e.dtype() {
        Dtype::F32 => 2f64.powi(24),
        Dtype::F64 => 2f64.powi(53),
    });
    let below_limit = e.compare("LT", &draw, &limit);
    let nonnegative = e.compare("GE", &draw, &zero);
    let valid = e.and(&valid, &gate_valid);
    let valid = e.and(&valid, &below_limit);
    let valid = e.and(&valid, &nonnegative);
    e.record_parameter_validity(&valid);
    let invalid = e.scalar(-1.0);
    Ok(e.select(&valid, &draw, &invalid))
}

fn lane_type(e: &Emitter) -> MlirTy {
    e.batch_shape().map_or(MlirTy::Scalar, |dims| {
        MlirTy::Ranked(dims.into_iter().map(Some).collect())
    })
}

fn param(e: &mut Emitter, params: &Params, name: &str, ty: &MlirTy) -> Result<Value, EmitError> {
    let value = params.get(e, name)?;
    let value = e.convert(&value, ElemKind::Real);
    if value.ty == MlirTy::Scalar && *ty != MlirTy::Scalar {
        Ok(e.broadcast_in_dim(&value, &[], ty.clone()))
    } else {
        Ok(value)
    }
}

/// The quantile of an exponential of rate one, cut off at `end`.
fn exponential_before(e: &mut Emitter, uniform: &Value, end: &Value) -> Value {
    let upper = e.neg(end);
    let signed_mass = e.expm1(&upper);
    let fraction = e.mul(uniform, &signed_mass);
    let log_survival = e.log1p(&fraction);
    e.neg(&log_survival)
}

/// For a unit-rate Poisson process conditioned to have an arrival by time
/// `rate`, draw that first arrival and independently count the rest.
fn positive_poisson(e: &mut Emitter, rate: &Value) -> Value {
    let uniform = e.rng("UNIFORM", &rate.ty);
    let first = exponential_before(e, &uniform, rate);
    let remaining = e.sub(rate, &first);
    let zero = e.scalar(0.0);
    let remaining = e.max(&remaining, &zero);
    let later = draw_poisson(e, &remaining);
    let one = e.scalar(1.0);
    let draw = e.add(&one, &later);
    let valid = e.compare("GE", &later, &zero);
    e.select(&valid, &draw, &later)
}

/// NB is compound Poisson with rate alpha * log(1 + 1/beta) and logarithmic
/// summands. Conditioning on positive count conditions only the number of
/// summands. When p(0) <= 1/2, Gamma-Poisson rejection needs at most two draws
/// in expectation and avoids a sum whose length grows with the rate.
fn positive_negative_binomial(
    e: &mut Emitter,
    alpha: &Value,
    beta: &Value,
    active: &Value,
) -> Value {
    let zero = e.constant(0.0, alpha.ty.clone());
    let log_beta = e.log(beta);
    let log_inverse_beta = e.neg(&log_beta);
    let magnitude = e.abs(&log_inverse_beta);
    let negative_magnitude = e.neg(&magnitude);
    let small = e.exp(&negative_magnitude);
    let correction = e.log1p(&small);
    let offset = e.max(&log_inverse_beta, &zero);
    let log1p_inverse_beta = e.add(&offset, &correction);
    let rate = e.mul(alpha, &log1p_inverse_beta);
    let cutoff = e.scalar(std::f64::consts::LN_2);
    let use_rejection = e.compare("GE", &rate, &cutoff);
    let reject_active = e.and(active, &use_rejection);
    let logseries_active = e.not(&use_rejection);
    let logseries_active = e.and(active, &logseries_active);
    let cluster_rate = e.select(&logseries_active, &rate, &zero);
    let clusters = positive_poisson(e, &cluster_rate);
    let count = e.select(&logseries_active, &clusters, &zero);
    let log_success = e.neg(&log1p_inverse_beta);
    let sum = logarithmic_sum(e, &count, &log_success);
    let rejection = reject_zero_negative_binomial(e, alpha, beta, &reject_active);
    e.select(&use_rejection, &rejection, &sum)
}

fn reject_zero_negative_binomial(
    e: &mut Emitter,
    alpha: &Value,
    beta: &Value,
    active: &Value,
) -> Value {
    let zero = e.constant(0.0, alpha.ty.clone());
    let one = e.scalar(1.0);
    let done = e.not(active);
    rejection_draw(e, done, zero.clone(), |e, done| {
        // Closed lanes must not drive the inner Poisson sampler's work.
        let alpha = e.select(done, &one, alpha);
        let beta = e.select(done, &one, beta);
        let lambda = draw_gamma(e, &alpha, &beta);
        let lambda = e.select(done, &zero, &lambda);
        let draw = draw_poisson(e, &lambda);
        // A negative result from draw_poisson means its rate overflowed.
        let accepted = e.compare("NE", &draw, &zero);
        (draw, accepted)
    })
}

fn logarithmic_sum(e: &mut Emitter, count: &Value, log_success: &Value) -> Value {
    let zero = e.constant(0.0, count.ty.clone());
    let one = e.scalar(1.0);
    let real_ty = count.ty.render(e.dtype(), ElemKind::Real);
    let types = [
        real_ty.clone(),
        real_ty,
        MlirTy::Key.render(e.dtype(), ElemKind::Real),
    ];
    let results = e.while_loop(
        &[count.clone(), zero.clone(), e.cur_key()],
        &types,
        |e, args| {
            let done = e.compare("LE", &args[0], &zero);
            poisson_pending(e, &done)
        },
        |e, args| {
            e.set_cur_key(args[2].clone());
            let active = e.compare("GT", &args[0], &zero);
            let draw = logarithmic(e, log_success);
            let draw = e.select(&active, &draw, &zero);
            let sum = e.add(&args[1], &draw);
            let remaining = e.sub(&args[0], &one);
            let remaining = e.max(&remaining, &zero);
            vec![remaining, sum, e.cur_key()]
        },
    );
    e.set_cur_key(results[2].clone());
    results[1].clone()
}

/// Kemp (1981), algorithm LK. Compare in log space because q can round
/// to one when beta is small, while log(q) is still representable.
fn logarithmic(e: &mut Emitter, log_success: &Value) -> Value {
    let one = e.scalar(1.0);
    let two = e.scalar(2.0);
    let u = e.rng("UNIFORM", &log_success.ty);
    let v = e.rng("UNIFORM", &log_success.ty);
    let u = e.sub(&one, &u);
    let v = e.sub(&one, &v);
    let log_v = e.log(&v);
    let log_q = e.special_log1mexp(log_success);
    let scaled = e.mul(&u, log_success);
    let log_threshold = e.special_log1mexp(&scaled);
    let log_threshold_sq = e.mul(&two, &log_threshold);
    let invert = e.compare("LE", &log_v, &log_threshold_sq);
    let inverse = e.div(&log_v, &log_threshold);
    let inverse = e.floor(&inverse);
    let inverse = e.add(&one, &inverse);
    let above_threshold = e.compare("GE", &log_v, &log_threshold);
    let small = e.select(&above_threshold, &one, &two);
    let conditional = e.select(&invert, &inverse, &small);
    let above_q = e.compare("GE", &log_v, &log_q);
    e.select(&above_q, &one, &conditional)
}

fn normal(e: &mut Emitter, params: &Params) -> Result<Value, EmitError> {
    let ty = lane_type(e);
    let mu = param(e, params, "mu", &ty)?;
    let sigma = param(e, params, "sigma", &ty)?;
    let lo = param(e, params, "lower", &ty)?;
    let hi = param(e, params, "upper", &ty)?;
    let zero = e.constant(0.0, ty.clone());
    let one = e.scalar(1.0);
    let infinity = e.scalar(f64::INFINITY);
    let lo_shift = e.sub(&lo, &mu);
    let hi_shift = e.sub(&hi, &mu);
    let lower = e.div(&lo_shift, &sigma);
    let upper = e.div(&hi_shift, &sigma);
    let width = e.sub(&hi, &lo);
    let span = e.div(&width, &sigma);
    let left = e.compare("LE", &hi, &mu);
    let right = e.compare("GE", &lo, &mu);
    let tail = e.or(&left, &right);
    let crosses = e.not(&tail);
    let broad = e.compare("GE", &span, &one);
    let central = e.and(&crosses, &broad);

    // Reflect lower tails. For z = a + t measured from the nearer bound,
    // propose t from an exponential of rate r = max(a, 1), truncated at
    // the far bound. The log target/proposal ratio up to a constant is
    // -(t - (r - a))^2 / 2, maximized at min(span, r - a).
    let reflected = e.neg(&upper);
    let a = e.select(&left, &reflected, &lower);
    let a = e.select(&tail, &a, &zero);
    let rate = e.max(&a, &one);
    let center = e.sub(&rate, &a);
    let mode = e.min(&span, &center);
    let end = e.mul(&rate, &span);

    let ordered = e.compare("LT", &lo, &hi);
    let positive_sigma = e.compare("GT", &sigma, &zero);
    let finite_sigma = e.compare("LT", &sigma, &infinity);
    let magnitude = e.abs(&mu);
    let finite_mu = e.compare("LT", &magnitude, &infinity);
    let representable = e.compare("LT", &a, &infinity);
    let valid = e.and(&ordered, &positive_sigma);
    let valid = e.and(&valid, &finite_sigma);
    let valid = e.and(&valid, &finite_mu);
    let valid = e.and(&valid, &representable);
    e.record_parameter_validity(&valid);

    let done = e.not(&valid);
    let drawn = rejection_draw(e, done, zero, |e, _| {
        let z = e.rng("NORMAL", &ty);
        let uniform = e.rng("UNIFORM", &ty);
        let acceptance = e.rng("UNIFORM", &ty);
        let log_u = e.log(&acceptance);
        let minus_half = e.scalar(-0.5);
        let two = e.scalar(2.0);

        let distance = exponential_before(e, &uniform, &end);
        let distance = e.div(&distance, &rate);
        let dx = e.mul(&distance, &sigma);
        let mirrored = e.sub(&hi, &dx);
        let ordinary = e.add(&lo, &dx);
        let tail_value = e.select(&left, &mirrored, &ordinary);
        let centered = e.sub(&distance, &mode);
        let sum = e.add(&distance, &mode);
        let twice_center = e.mul(&two, &center);
        let other = e.sub(&sum, &twice_center);
        let log_ratio = e.mul(&centered, &other);
        let log_ratio = e.mul(&minus_half, &log_ratio);
        let tail_accept = e.compare("LE", &log_u, &log_ratio);

        // Narrow intervals containing zero use a uniform proposal, whose
        // likelihood ratio is bounded by the density at z = 0.
        let fraction = e.mul(&width, &uniform);
        let narrow_value = e.add(&lo, &fraction);
        let narrow_shift = e.sub(&narrow_value, &mu);
        let narrow_z = e.div(&narrow_shift, &sigma);
        let squared = e.mul(&narrow_z, &narrow_z);
        let narrow_log_ratio = e.mul(&minus_half, &squared);
        let narrow_accept = e.compare("LE", &log_u, &narrow_log_ratio);

        // Wider intervals containing zero accept an ordinary Normal draw
        // falling in the interval. Their acceptance probability stays bounded.
        let above = e.compare("GE", &z, &lower);
        let below = e.compare("LE", &z, &upper);
        // The probit RNG can produce -inf when its uniform bits are zero.
        let magnitude = e.abs(&z);
        let finite = e.compare("LT", &magnitude, &infinity);
        let central_accept = e.and(&above, &below);
        let central_accept = e.and(&central_accept, &finite);
        let scaled = e.mul(&sigma, &z);
        let central_value = e.add(&mu, &scaled);
        let value = e.select(&tail, &tail_value, &narrow_value);
        let value = e.select(&central, &central_value, &value);
        let accepted = e.select(&tail, &tail_accept, &narrow_accept);
        let accepted = e.select(&central, &central_accept, &accepted);
        (value, accepted)
    });

    // Transforming back from standardized coordinates can round to an endpoint.
    let drawn = e.max(&lo, &drawn);
    let drawn = e.min(&hi, &drawn);
    let undefined = e.scalar(f64::NAN);
    Ok(e.select(&valid, &drawn, &undefined))
}

fn rejection_draw(
    e: &mut Emitter,
    done: Value,
    initial: Value,
    proposal: impl FnOnce(&mut Emitter, &Value) -> (Value, Value),
) -> Value {
    let types = [
        done.ty.render(e.dtype(), ElemKind::Bool),
        initial.ty.render(e.dtype(), ElemKind::Real),
        MlirTy::Key.render(e.dtype(), ElemKind::Real),
    ];
    let results = e.while_loop(
        &[done, initial, e.cur_key()],
        &types,
        |e, args| poisson_pending(e, &args[0]),
        |e, args| {
            e.set_cur_key(args[2].clone());
            let (draw, accepted) = proposal(e, &args[0]);
            let accepted_draw = e.select(&accepted, &draw, &args[1]);
            let draw = e.select(&args[0], &args[1], &accepted_draw);
            let done = e.or(&args[0], &accepted);
            vec![done, draw, e.cur_key()]
        },
    );
    e.set_cur_key(results[2].clone());
    results[1].clone()
}
