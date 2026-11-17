//! Direct draws of the normalized §09 HEP laws. Rejection carries its key and
//! reports exhaustion; fixed inversions retain a certified probability bracket.

use super::*;

// Chambers–Mallows–Stuck, JASA 71 (1976), doi:10.1080/01621459.1976.10480344.
// This alpha=1, beta=1 transform has the §09 Landau convention directly.
pub(crate) fn landau_sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
    let loc = parameter(e, p, "loc", ParameterDomain::Real)?;
    let scale = parameter(e, p, "scale", ParameterDomain::Positive)?;
    let u = open_uniform(e, &loc);
    let v = open_uniform(e, &loc);
    let pi = e.scalar(std::f64::consts::PI);
    let angle = e.mul(&pi, &u);
    let half_pi = e.scalar(std::f64::consts::FRAC_PI_2);
    let centered = e.sub(&angle, &half_pi);
    let tangent = e.tan(&centered);
    let first = e.mul(&angle, &tangent);
    let log_v = e.log(&v);
    let exponential = e.neg(&log_v);
    let cosine = e.cos(&centered);
    let numerator = e.mul(&exponential, &cosine);
    let ratio = e.div(&numerator, &angle);
    let logarithm = e.log(&ratio);
    let standard = e.sub(&first, &logarithm);
    let offset = e.mul(&scale, &standard);
    Ok(e.add(&loc, &offset))
}

pub(crate) fn voigtian_sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "mean", ParameterDomain::Real)?;
    let width = parameter(e, p, "width", ParameterDomain::Positive)?;
    let sigma = parameter(e, p, "sigma", ParameterDomain::Positive)?;
    let u = open_uniform(e, &mean);
    let v = open_uniform(e, &mean);
    let gaussian = e.probit(&u);
    let gaussian = e.mul(&sigma, &gaussian);
    let half = e.scalar(0.5);
    let centered = e.sub(&v, &half);
    let pi = e.scalar(std::f64::consts::PI);
    let angle = e.mul(&pi, &centered);
    let tangent = e.tan(&angle);
    let gamma = e.mul(&half, &width);
    let cauchy = e.mul(&gamma, &tangent);
    let offset = e.add(&gaussian, &cauchy);
    Ok(e.add(&mean, &offset))
}

pub(crate) fn bifurcated_normal_sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "mean", ParameterDomain::Real)?;
    let sigma_left = parameter(e, p, "sigmaL", ParameterDomain::Positive)?;
    let sigma_right = parameter(e, p, "sigmaR", ParameterDomain::Positive)?;
    let u = open_uniform(e, &mean);
    let v = open_uniform(e, &mean);
    let scale = e.max(&sigma_left, &sigma_right);
    let left_weight = e.div(&sigma_left, &scale);
    let right_weight = e.div(&sigma_right, &scale);
    let total = e.add(&left_weight, &right_weight);
    let choice = e.mul(&u, &total);
    let on_left = e.compare("LT", &choice, &left_weight);
    let minus_left = e.neg(&sigma_left);
    let signed_sigma = e.select(&on_left, &minus_left, &sigma_right);
    let half = e.scalar(0.5);
    let probability = e.mul(&half, &v);
    let z = e.probit(&probability);
    let magnitude = e.neg(&z);
    let offset = e.mul(&signed_sigma, &magnitude);
    Ok(e.add(&mean, &offset))
}

fn log_hypot(e: &mut Emitter, x: &Value, y: &Value) -> Value {
    let ax = e.abs(x);
    let ay = e.abs(y);
    let scale = e.max(&ax, &ay);
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let nonzero = e.compare("GT", &scale, &zero);
    let safe = e.select(&nonzero, &scale, &one);
    let xr = e.div(x, &safe);
    let yr = e.div(y, &safe);
    let xx = e.mul(&xr, &xr);
    let yy = e.mul(&yr, &yr);
    let square = e.add(&xx, &yy);
    let square = e.select(&nonzero, &square, &one);
    let logarithm = e.log(&square);
    let half = e.scalar(0.5);
    let logarithm = e.mul(&half, &logarithm);
    let log_scale = e.log(&safe);
    e.add(&log_scale, &logarithm)
}

// With x=sqrt(mean*hypot(mean,width))*t, the positive-halfline density
// is proportional to 1/(t^4-2*eta*t^2+1). Factoring that quartic into
// ((t-a/2)^2+d^2)*((t+a/2)^2+d^2) gives this normalized antiderivative.
fn breit_wigner_cdf(e: &mut Emitter, t: &Value, a: &Value, d: &Value) -> Value {
    let one = e.scalar(1.0);
    let two = e.scalar(2.0);
    let half = e.scalar(0.5);
    let t2 = e.mul(t, t);
    let denominator = e.sub(&one, &t2);
    let numerator = e.mul(t, d);
    let numerator = e.mul(&two, &numerator);
    let angle = e.atan2(&numerator, &denominator);
    let half_a = e.mul(&half, a);
    let gap = e.sub(t, &half_a);
    let log_minus = log_hypot(e, &gap, d);
    let log_minus = e.mul(&two, &log_minus);
    let twice_a = e.mul(&two, a);
    let log_a = e.log(&twice_a);
    let log_t = e.log(t);
    let ratio = e.add(&log_a, &log_t);
    let ratio = e.sub(&ratio, &log_minus);
    let logarithm = softplus(e, &ratio);
    let fraction = e.div(d, a);
    let correction = e.mul(&fraction, &logarithm);
    let value = e.add(&angle, &correction);
    let pi = e.scalar(std::f64::consts::PI);
    let value = e.div(&value, &pi);
    let zero = e.scalar(0.0);
    let value = e.max(&value, &zero);
    e.min(&value, &one)
}

pub(crate) fn relativistic_breit_wigner_sample(
    e: &mut Emitter,
    p: &Params,
) -> Result<Value, EmitError> {
    let mean = parameter(e, p, "mean", ParameterDomain::Positive)?;
    let width = parameter(e, p, "width", ParameterDomain::Positive)?;
    let maximum = e.max(&mean, &width);
    let mr = e.div(&mean, &maximum);
    let wr = e.div(&width, &maximum);
    let mr2 = e.mul(&mr, &mr);
    let wr2 = e.mul(&wr, &wr);
    let norm = e.add(&mr2, &wr2);
    let norm = e.sqrt(&norm);
    let hypot = e.mul(&maximum, &norm);
    let infinity = e.inf_like(&hypot);
    let finite = e.compare("LT", &hypot, &infinity);
    e.record_runtime_error(&finite, 10);
    let hypot = e.select(&finite, &hypot, &maximum);
    let eta = e.div(&mean, &hypot);
    let beta = e.div(&width, &hypot);
    let one = e.scalar(1.0);
    let two = e.scalar(2.0);
    let a = e.add(&one, &eta);
    let a = e.mul(&two, &a);
    let a = e.sqrt(&a);
    let d = e.div(&beta, &a);
    let root_mean = e.sqrt(&mean);
    let root_hypot = e.sqrt(&hypot);
    let physical_scale = e.mul(&root_mean, &root_hypot);
    let finite = e.compare("LT", &physical_scale, &infinity);
    e.record_runtime_error(&finite, 10);
    let physical_scale = e.select(&finite, &physical_scale, &mean);
    let probability = open_uniform(e, &mean);
    let axes = e.axes_of(&probability);
    let upper = e.constant_like(2.0, &probability);
    let result = e.matrix_loop(32, 0, &[upper], |e, _, state| {
        let upper = e.axes_view(&state[0], axes.clone());
        let cdf = breit_wigner_cdf(e, &upper, &a, &d);
        let extend = e.compare("LT", &cdf, &probability);
        let doubled = e.mul(&two, &upper);
        vec![e.select(&extend, &doubled, &upper)]
    });
    let upper = e.axes_view(&result[0], axes.clone());
    let cdf = breit_wigner_cdf(e, &upper, &a, &d);
    let bracketed = e.compare("GE", &cdf, &probability);
    e.record_runtime_error(&bracketed, 10);
    let lower = e.constant_like(0.0, &probability);
    let half = e.scalar(0.5);
    let result = e.matrix_loop(64, 0, &[lower, upper], |e, _, state| {
        let lower = e.axes_view(&state[0], axes.clone());
        let upper = e.axes_view(&state[1], axes.clone());
        let sum = e.add(&lower, &upper);
        let middle = e.mul(&half, &sum);
        let cdf = breit_wigner_cdf(e, &middle, &a, &d);
        let below = e.compare("LT", &cdf, &probability);
        vec![
            e.select(&below, &middle, &lower),
            e.select(&below, &upper, &middle),
        ]
    });
    let lower = e.axes_view(&result[0], axes.clone());
    let upper = e.axes_view(&result[1], axes);
    let sum = e.add(&lower, &upper);
    let quantile = e.mul(&half, &sum);
    // A final Newton correction retains the implicit CDF derivative: bisection's
    // branch decisions alone have zero derivative with respect to a and d.
    let cdf = breit_wigner_cdf(e, &quantile, &a, &d);
    let residual = e.sub(&cdf, &probability);
    let half_a = e.mul(&half, &a);
    let minus = e.sub(&quantile, &half_a);
    let plus = e.add(&quantile, &half_a);
    let minus2 = e.mul(&minus, &minus);
    let plus2 = e.mul(&plus, &plus);
    let d2 = e.mul(&d, &d);
    let first = e.add(&minus2, &d2);
    let second = e.add(&plus2, &d2);
    let quartic = e.mul(&first, &second);
    let pi_over_four = e.scalar(std::f64::consts::FRAC_PI_4);
    let inverse_density = e.mul(&quartic, &pi_over_four);
    let inverse_density = e.div(&inverse_density, &d);
    let correction = e.mul(&residual, &inverse_density);
    let quantile = e.sub(&quantile, &correction);
    Ok(e.mul(&physical_scale, &quantile))
}

// t=1-(x/resonance)^2 has density proportional to t^power*exp(slope*t).
// Strong negative slopes use a gamma draw conditioned on t<1. The other
// envelopes are a power law (negative/weak slope) or truncated exponential
// in 1-t (strong positive slope); their log acceptance ratios stay <=0.
fn argus_proposal(e: &mut Emitter, power: &Value, slope: &Value, u: &Value) -> (Value, Value) {
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let half = e.scalar(0.5);
    let shape = e.add(power, &one);
    let rate = e.neg(slope);
    let root = e.sqrt(&shape);
    let central = e.sub(&shape, &root);
    let half_shape = e.mul(&half, &shape);
    let threshold = e.max(&central, &half_shape);
    let negative = e.compare("LT", slope, &zero);
    let central_gamma = e.compare("GE", &rate, &threshold);
    let gamma_mode = e.and(&negative, &central_gamma);
    let gamma_shape = e.select(&gamma_mode, &shape, &one);
    let gamma_rate = e.select(&gamma_mode, &rate, &one);
    let gamma = draw_gamma(e, &gamma_shape, &gamma_rate);
    let gamma_positive = e.compare("GT", &gamma, &zero);
    let gamma_below = e.compare("LT", &gamma, &one);
    let gamma_accepted = e.and(&gamma_positive, &gamma_below);
    let gamma_distance = e.sub(&one, &gamma);
    let not_gamma = e.not(&gamma_mode);
    let negative_power = e.and(&negative, &not_gamma);
    let exponent = e.add(&shape, slope);
    let exponent = e.select(&negative_power, &exponent, &shape);
    let logarithm = e.log(u);
    let log_t = e.div(&logarithm, &exponent);
    let distance = e.expm1(&log_t);
    let power_distance = e.neg(&distance);
    let exp_mode = e.compare("GT", slope, &shape);
    let positive_rate = e.select(&exp_mode, slope, &one);
    let negative_rate = e.neg(&positive_rate);
    let survival = e.expm1(&negative_rate);
    let survival = e.neg(&survival);
    let uniform_mass = e.mul(u, &survival);
    let negative_mass = e.neg(&uniform_mass);
    let exponential_distance = e.log1p(&negative_mass);
    let exponential_distance = e.neg(&exponential_distance);
    let exponential_distance = e.div(&exponential_distance, &positive_rate);
    let distance = e.select(&exp_mode, &exponential_distance, &power_distance);
    let minus_distance = e.neg(&exponential_distance);
    let log_t_exp = e.log1p(&minus_distance);
    let exp_accept = e.mul(power, &log_t_exp);
    let power_accept = e.mul(slope, &power_distance);
    let power_accept = e.neg(&power_accept);
    let remainder = expm1_minus_x(e, &log_t);
    let negative_accept = e.mul(slope, &remainder);
    let power_accept = e.select(&negative_power, &negative_accept, &power_accept);
    let log_accept = e.select(&exp_mode, &exp_accept, &power_accept);
    let v = open_uniform(e, u);
    let log_v = e.log(&v);
    let accepted = e.compare("LE", &log_v, &log_accept);
    (
        e.select(&gamma_mode, &gamma_distance, &distance),
        e.select(&gamma_mode, &gamma_accepted, &accepted),
    )
}

pub(crate) fn argus_sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
    let resonance = parameter(e, p, "resonance", ParameterDomain::Positive)?;
    let slope = parameter(e, p, "slope", ParameterDomain::Real)?;
    let power = parameter(e, p, "power", ParameterDomain::Positive)?;
    let u = open_uniform(e, &resonance);
    let axes = e.axes_of(&u);
    let (candidate, done) = argus_proposal(e, &power, &slope, &u);
    let zero = e.constant_like(0.0, &u);
    let initial = e.select(&done, &candidate, &zero);
    let index = e.int_value_const(1);
    let types = [
        done.ty.render(e.dtype(), ElemKind::Bool),
        initial.ty.render(e.dtype(), ElemKind::Real),
        index.ty.render(e.dtype(), ElemKind::Int),
        MlirTy::Key.render(e.dtype(), ElemKind::Real),
    ];
    let result = e.while_loop(
        &[done, initial, index, e.cur_key()],
        &types,
        |e, state| {
            let pending = poisson_pending(e, &state[0]);
            let limit = e.int_value_const(1024);
            let within = e.compare("LT", &state[2], &limit);
            e.and(&pending, &within)
        },
        |e, state| {
            e.set_cur_key(state[3].clone());
            let done = e.axes_view(&state[0], axes.clone());
            let previous = e.axes_view(&state[1], axes.clone());
            let u = open_uniform(e, &resonance);
            let (candidate, accepted) = argus_proposal(e, &power, &slope, &u);
            let pending = e.not(&done);
            let take = e.and(&pending, &accepted);
            let value = e.select(&take, &candidate, &previous);
            let done = e.or(&done, &accepted);
            let one = e.int_value_const(1);
            let index = e.add(&state[2], &one);
            vec![done, value, index, e.cur_key()]
        },
    );
    e.set_cur_key(result[3].clone());
    let done = e.axes_view(&result[0], axes.clone());
    e.record_runtime_error(&done, 10);
    let distance = e.axes_view(&result[1], axes);
    let root = e.sqrt(&distance);
    Ok(e.mul(&resonance, &root))
}
