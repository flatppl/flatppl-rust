//! Exact Gaussian-core / Pareto-tail normalization and piecewise inversion.

use super::*;

struct Side {
    alpha: Value,
    n: Value,
    log_tail: Value,
    log_core: Value,
    half_core: Value,
}

fn side(e: &mut Emitter, p: &Params, alpha: &str, n: &str) -> Result<Side, EmitError> {
    let alpha = parameter(e, p, alpha, ParameterDomain::Positive)?;
    let n = parameter(e, p, n, ParameterDomain::TailExponent)?;
    let log_alpha = e.log(&alpha);
    let log_n = e.log(&n);
    let one = e.scalar(1.0);
    let n_minus_one = e.sub(&n, &one);
    let log_n_minus_one = e.log(&n_minus_one);
    let square = e.mul(&alpha, &alpha);
    let half = e.scalar(0.5);
    let half_square = e.mul(&half, &square);
    let log_tail = e.sub(&log_n, &log_alpha);
    let log_tail = e.sub(&log_tail, &log_n_minus_one);
    let log_tail = e.sub(&log_tail, &half_square);
    let root_two = e.scalar(std::f64::consts::SQRT_2);
    let argument = e.div(&alpha, &root_two);
    let erf = e.erf(&argument);
    let half_integral = e.scalar((std::f64::consts::PI / 2.0).sqrt());
    let half_core = e.mul(&half_integral, &erf);
    let log_core = e.log(&half_core);
    Ok(Side {
        alpha,
        n,
        log_tail,
        log_core,
        half_core,
    })
}

fn kernel(e: &mut Emitter, z: &Value, u: &Value, alpha: &Value, n: &Value) -> Value {
    let in_tail = e.compare("GT", u, alpha);
    let one = e.scalar(1.0);
    let zero = e.scalar(0.0);
    let delta = e.sub(u, alpha);
    let safe_delta = e.select(&in_tail, &delta, &one);
    let log_delta = e.log(&safe_delta);
    let log_alpha = e.log(alpha);
    let log_n = e.log(n);
    let ratio = e.add(&log_delta, &log_alpha);
    let ratio = e.sub(&ratio, &log_n);
    let logarithm = softplus(e, &ratio);
    let power = e.mul(n, &logarithm);
    let square = e.mul(alpha, alpha);
    let minus_half = e.scalar(-0.5);
    let boundary = e.mul(&minus_half, &square);
    let tail = e.sub(&boundary, &power);
    let core_z = e.select(&in_tail, &zero, z);
    let square = e.mul(&core_z, &core_z);
    let core = e.mul(&minus_half, &square);
    e.select(&in_tail, &tail, &core)
}

pub(crate) fn crystal_ball_logpdf(
    e: &mut Emitter,
    p: &Params,
    x: &Value,
) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "m0", ParameterDomain::Real)?;
    let sigma = parameter(e, p, "sigma", ParameterDomain::Positive)?;
    let side = side(e, p, "alpha", "n")?;
    let difference = e.sub(x, &mean);
    let z = e.div(&difference, &sigma);
    let u = e.neg(&z);
    let score = kernel(e, &z, &u, &side.alpha, &side.n);
    let full_half = e.scalar((std::f64::consts::PI / 2.0).sqrt());
    let core = e.add(&side.half_core, &full_half);
    let log_core = e.log(&core);
    let log_total = logadd(e, &side.log_tail, &log_core);
    let log_sigma = e.log(&sigma);
    let normalized = e.sub(&score, &log_total);
    Ok(e.sub(&normalized, &log_sigma))
}

pub(crate) fn crystal_ball_sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "m0", ParameterDomain::Real)?;
    let sigma = parameter(e, p, "sigma", ParameterDomain::Positive)?;
    let side = side(e, p, "alpha", "n")?;
    let full_half = e.scalar((std::f64::consts::PI / 2.0).sqrt());
    let core = e.add(&side.half_core, &full_half);
    let log_core = e.log(&core);
    let log_total = logadd(e, &side.log_tail, &log_core);
    let log_fraction = e.sub(&side.log_tail, &log_total);
    let draw = open_uniform(e, &mean);
    let log_draw = e.log(&draw);
    let tail = e.compare("LT", &log_draw, &log_fraction);
    let one = e.scalar(1.0);
    let n_minus_one = e.sub(&side.n, &one);
    let exponent = e.sub(&log_fraction, &log_draw);
    let exponent = e.div(&exponent, &n_minus_one);
    let offset = e.expm1(&exponent);
    let ratio = e.div(&side.n, &side.alpha);
    let offset = e.mul(&ratio, &offset);
    let tail_z = e.add(&side.alpha, &offset);
    let tail_z = e.neg(&tail_z);
    let minus_draw = e.neg(&draw);
    let log_survival = e.log1p(&minus_draw);
    let log_probability = e.add(&log_survival, &log_total);
    let log_root_two_pi = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
    let log_probability = e.sub(&log_probability, &log_root_two_pi);
    let probability = e.exp(&log_probability);
    let half = e.scalar(0.5);
    let probability = e.select(&tail, &half, &probability);
    let core_z = e.probit(&probability);
    let core_z = e.neg(&core_z);
    let z = e.select(&tail, &tail_z, &core_z);
    let offset = e.mul(&sigma, &z);
    Ok(e.add(&mean, &offset))
}

fn double_weights(
    e: &mut Emitter,
    left: &Side,
    right: &Side,
    sigma_left: &Value,
    sigma_right: &Value,
) -> [Value; 4] {
    let log_left = e.log(sigma_left);
    let log_right = e.log(sigma_right);
    [
        e.add(&log_left, &left.log_tail),
        e.add(&log_left, &left.log_core),
        e.add(&log_right, &right.log_core),
        e.add(&log_right, &right.log_tail),
    ]
}

pub(crate) fn double_sided_crystal_ball_logpdf(
    e: &mut Emitter,
    p: &Params,
    x: &Value,
) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "m0", ParameterDomain::Real)?;
    let sigma_left = parameter(e, p, "sigmaL", ParameterDomain::Positive)?;
    let sigma_right = parameter(e, p, "sigmaR", ParameterDomain::Positive)?;
    let left = side(e, p, "alphaL", "nL")?;
    let right = side(e, p, "alphaR", "nR")?;
    let weights = double_weights(e, &left, &right, &sigma_left, &sigma_right);
    let left_mass = logadd(e, &weights[0], &weights[1]);
    let right_mass = logadd(e, &weights[2], &weights[3]);
    let log_total = logadd(e, &left_mass, &right_mass);
    let on_left = e.compare("LT", x, &mean);
    let sigma = e.select(&on_left, &sigma_left, &sigma_right);
    let alpha = e.select(&on_left, &left.alpha, &right.alpha);
    let n = e.select(&on_left, &left.n, &right.n);
    let difference = e.sub(x, &mean);
    let z = e.div(&difference, &sigma);
    let u = e.abs(&z);
    let score = kernel(e, &z, &u, &alpha, &n);
    Ok(e.sub(&score, &log_total))
}

fn tail_quantile(e: &mut Emitter, v: &Value, side: &Side) -> Value {
    let logarithm = e.log(v);
    let logarithm = e.neg(&logarithm);
    let one = e.scalar(1.0);
    let denominator = e.sub(&side.n, &one);
    let exponent = e.div(&logarithm, &denominator);
    let exponential = e.expm1(&exponent);
    let ratio = e.div(&side.n, &side.alpha);
    let tail = e.mul(&ratio, &exponential);
    e.add(&side.alpha, &tail)
}

fn half_core_quantile(e: &mut Emitter, v: &Value, side: &Side) -> Value {
    let normalizer = e.scalar((2.0 * std::f64::consts::PI).sqrt());
    let probability = e.mul(v, &side.half_core);
    let probability = e.div(&probability, &normalizer);
    let half = e.scalar(0.5);
    let probability = e.sub(&half, &probability);
    let quantile = e.probit(&probability);
    e.neg(&quantile)
}

pub(crate) fn double_sided_crystal_ball_sample(
    e: &mut Emitter,
    p: &Params,
) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "m0", ParameterDomain::Real)?;
    let sigma_left = parameter(e, p, "sigmaL", ParameterDomain::Positive)?;
    let sigma_right = parameter(e, p, "sigmaR", ParameterDomain::Positive)?;
    let left = side(e, p, "alphaL", "nL")?;
    let right = side(e, p, "alphaR", "nR")?;
    let weights = double_weights(e, &left, &right, &sigma_left, &sigma_right);
    let high_left = e.max(&weights[0], &weights[1]);
    let high_right = e.max(&weights[2], &weights[3]);
    let high = e.max(&high_left, &high_right);
    let weights: Vec<_> = weights
        .iter()
        .map(|w| {
            let shifted = e.sub(w, &high);
            e.exp(&shifted)
        })
        .collect();
    let first_two = e.add(&weights[0], &weights[1]);
    let first_three = e.add(&first_two, &weights[2]);
    let total = e.add(&first_three, &weights[3]);
    let draw = open_uniform(e, &mean);
    let choice = e.mul(&draw, &total);
    let v = open_uniform(e, &mean);
    let in_left_tail = e.compare("LT", &choice, &weights[0]);
    let in_left_core = e.compare("LT", &choice, &first_two);
    let in_right_core = e.compare("LT", &choice, &first_three);
    let in_right_tail = e.not(&in_right_core);
    // Near n=1, an unselected Pareto quantile may overflow. Its inactive
    // input must be safe before expm1 so reverse AD never multiplies 0 by inf.
    let one = e.scalar(1.0);
    let left_v = e.select(&in_left_tail, &v, &one);
    let right_v = e.select(&in_right_tail, &v, &one);
    let left_tail = tail_quantile(e, &left_v, &left);
    let left_core = half_core_quantile(e, &v, &left);
    let right_core = half_core_quantile(e, &v, &right);
    let right_tail = tail_quantile(e, &right_v, &right);
    let left_z = e.select(&in_left_tail, &left_tail, &left_core);
    let left_offset = e.mul(&sigma_left, &left_z);
    let left_value = e.sub(&mean, &left_offset);
    let right_z = e.select(&in_right_core, &right_core, &right_tail);
    let right_offset = e.mul(&sigma_right, &right_z);
    let right_value = e.add(&mean, &right_offset);
    Ok(e.select(&in_left_core, &left_value, &right_value))
}
