use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};
use std::f64::consts::{PI, TAU};

pub(crate) fn score(
    b: &mut Builder<'_>,
    response: NodeId,
    decision: NodeId,
    drift: NodeId,
    separation: NodeId,
    nondecision: NodeId,
    bias: NodeId,
) -> NodeId {
    let zero = b.lit_real(0.0);
    let half = b.lit_real(0.5);
    let one = b.lit_real(1.0);
    let cut = b.lit_real(0.125);
    let lower_code = b.lit_int(0);
    let upper_code = b.lit_int(1);
    let lower = b.call("equal", &[decision, lower_code]);
    let upper = b.call("equal", &[decision, upper_code]);
    let valid_decision = b.call("lor", &[lower, upper]);
    let time = b.call("sub", &[response, nondecision]);
    let positive_time = b.call("gt", &[time, zero]);
    let positive_separation = b.call("gt", &[separation, zero]);
    let nonnegative_delay = b.call("ge", &[nondecision, zero]);
    let positive_bias = b.call("gt", &[bias, zero]);
    let bias_below_one = b.call("lt", &[bias, one]);
    let valid_bias = b.call("land", &[positive_bias, bias_below_one]);
    let valid = [
        positive_time,
        positive_separation,
        nonnegative_delay,
        valid_bias,
        valid_decision,
    ]
    .into_iter()
    .fold(b.call("isfinite", &[drift]), |valid, part| {
        b.call("land", &[valid, part])
    });

    // Reflect the drift and starting point for hits at the upper boundary.
    // Compute both distances directly to preserve very small starting biases.
    let bias = b.call("ifelse", &[valid, bias, half]);
    let reflected = b.call("sub", &[one, bias]);
    let distance = b.call("ifelse", &[upper, reflected, bias]);
    let complement = b.call("ifelse", &[upper, bias, reflected]);
    let negative_drift = b.call("neg", &[drift]);
    let towards_boundary = b.call("ifelse", &[upper, drift, negative_drift]);
    let towards_boundary = b.call("ifelse", &[valid, towards_boundary, zero]);

    let time = b.call("ifelse", &[valid, time, one]);
    let separation = b.call("ifelse", &[valid, separation, one]);
    let scaled_time = b.call("divide", &[time, separation]);
    let scaled_time = b.call("divide", &[scaled_time, separation]);
    let positive_scaled_time = b.call("gt", &[scaled_time, zero]);
    let finite_scaled_time = b.call("isfinite", &[scaled_time]);
    let valid = b.call("land", &[valid, positive_scaled_time]);
    let valid = b.call("land", &[valid, finite_scaled_time]);
    let scaled_time = b.call("ifelse", &[valid, scaled_time, cut]);

    // Navarro and Fuss (2009), doi:10.1016/j.jmp.2009.02.003.
    // Both series remain well conditioned at the switch. Clamp unused
    // branches before evaluation so their derivatives stay finite.
    let short = b.call("le", &[scaled_time, cut]);
    let short_time = b.call("ifelse", &[short, scaled_time, cut]);
    let long_time = b.call("ifelse", &[short, cut, scaled_time]);
    let images = log_image_weight(b, short_time, distance, complement);
    let log_time = b.call("log", &[short_time]);
    let minus_three_halves = b.lit_real(-1.5);
    let sqrt_two_pi = b.lit_real(0.5 * TAU.ln());
    let image_scale = b.call("mul", &[minus_three_halves, log_time]);
    let image_scale = b.call("sub", &[image_scale, sqrt_two_pi]);

    let minus_half = b.lit_real(-0.5);
    let actual_distance = b.call("mul", &[separation, distance]);
    let root_time = b.call("sqrt", &[time]);
    let drift_time = b.call("mul", &[towards_boundary, root_time]);
    let short_root = b.call("ifelse", &[short, root_time, one]);
    let short_distance = b.call("ifelse", &[short, actual_distance, zero]);
    let short_drift = b.call("ifelse", &[short, drift_time, zero]);
    let standardized = b.call("divide", &[short_distance, short_root]);
    let residual = b.call("sub", &[standardized, short_drift]);
    let residual_square = b.call("mul", &[residual, residual]);
    let image_exponent = b.call("mul", &[minus_half, residual_square]);
    let image_score = b.call("add", &[images, image_scale]);
    let image_score = b.call("add", &[image_score, image_exponent]);

    let spectral = log_spectral_density(b, long_time, distance, complement);
    let long_drift = b.call("ifelse", &[short, zero, drift_time]);
    let long_speed = b.call("ifelse", &[short, zero, towards_boundary]);
    let drift_square = b.call("mul", &[long_drift, long_drift]);
    let drift_square = b.call("mul", &[minus_half, drift_square]);
    let tilt = b.call("mul", &[actual_distance, long_speed]);
    let long_score = b.call("add", &[spectral, drift_square]);
    let long_score = b.call("add", &[long_score, tilt]);

    let unit_score = b.call("ifelse", &[short, image_score, long_score]);
    let log_separation = b.call("log", &[separation]);
    let two = b.lit_real(2.0);
    let log_jacobian = b.call("mul", &[two, log_separation]);
    let score = b.call("sub", &[unit_score, log_jacobian]);
    let impossible = b.lit_real(f64::NEG_INFINITY);
    b.call("ifelse", &[valid, score, impossible])
}

pub(crate) fn expected_time(
    b: &mut Builder<'_>,
    drift: NodeId,
    separation: NodeId,
    nondecision: NodeId,
    bias: NodeId,
) -> NodeId {
    let names = ["drift", "separation", "nondecision", "bias"];
    let locals = names.map(|name| Ref {
        ns: RefNs::Local,
        name: b.sym(&format!("_{name}_")),
    });
    let [speed, width, delay, start] = locals.map(|local| b.m.alloc(Node::Ref(local)));
    let mean = mean_first_passage(b, speed, width, start);
    let response = b.call("add", &[delay, mean]);
    let inputs = names
        .into_iter()
        .zip(locals)
        .map(|(name, local)| (b.sym(name), local))
        .collect::<Vec<_>>();
    let head = CallHead::Builtin(b.sym("functionof"));
    let row = b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([response]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    b.call("broadcast", &[row, drift, separation, nondecision, bias])
}

fn mean_first_passage(
    b: &mut Builder<'_>,
    drift: NodeId,
    separation: NodeId,
    bias: NodeId,
) -> NodeId {
    // At the hit time, E[X] = a P(upper) = a*w + drift E[T].
    // Reflect the process to use a nonnegative scaled drift.
    let zero = b.lit_real(0.0);
    let half = b.lit_real(0.5);
    let one = b.lit_real(1.0);
    let two = b.lit_real(2.0);
    let reflected = b.call("lt", &[drift, zero]);
    let negative_drift = b.call("neg", &[drift]);
    let speed = b.call("ifelse", &[reflected, negative_drift, drift]);
    let complement = b.call("sub", &[one, bias]);
    let start = b.call("ifelse", &[reflected, complement, bias]);
    let gap = b.call("ifelse", &[reflected, bias, complement]);
    let x = b.call("mul", &[speed, separation]);
    let x = b.call("mul", &[two, x]);
    let cutoff = b.lit_real(1e-4);
    let small = b.call("lt", &[x, cutoff]);

    let x_small = b.call("ifelse", &[small, x, zero]);
    let width_product = b.call("mul", &[start, gap]);
    let asymmetry = b.call("sub", &[gap, start]);
    let sixth = b.lit_real(1.0 / 6.0);
    let first_order = b.call("mul", &[sixth, asymmetry]);
    let first_order = b.call("mul", &[first_order, x_small]);
    let minus_twelfth = b.lit_real(-1.0 / 12.0);
    let second_order = b.call("mul", &[minus_twelfth, width_product]);
    let second_order = b.call("mul", &[second_order, x_small]);
    let second_order = b.call("mul", &[second_order, x_small]);
    let correction = b.call("add", &[first_order, second_order]);
    let adjustment = b.call("add", &[one, correction]);
    let small_mean = b.call("mul", &[width_product, adjustment]);

    // Evaluate the hit probability of the farther boundary when close to an
    // endpoint, then subtract distances on that smaller probability scale.
    let x = b.call("ifelse", &[small, cutoff, x]);
    let minus_x = b.call("neg", &[x]);
    let normalizer = b.call("expm1", &[minus_x]);
    let x_start = b.call("mul", &[minus_x, start]);
    let x_gap = b.call("mul", &[minus_x, gap]);
    let upper = b.call("expm1", &[x_start]);
    let upper = b.call("divide", &[upper, normalizer]);
    let lower = b.call("expm1", &[x_gap]);
    let lower = b.call("divide", &[lower, normalizer]);
    let tilt = b.call("exp", &[x_start]);
    let lower = b.call("mul", &[tilt, lower]);
    let nearer_lower = b.call("le", &[start, half]);
    let advance = b.call("sub", &[upper, start]);
    let remaining = b.call("sub", &[gap, lower]);
    let advance = b.call("ifelse", &[nearer_lower, advance, remaining]);
    let expected = b.call("divide", &[advance, x]);
    let expected = b.call("mul", &[two, expected]);

    let expected = b.call("ifelse", &[small, small_mean, expected]);
    let area = b.call("mul", &[separation, separation]);
    b.call("mul", &[area, expected])
}

/// Log of two-boundary to one-boundary first-passage subdensity. The drift
/// tilt and the dimensional Jacobian cancel in the ratio.
pub(crate) fn log_proposal_ratio(
    b: &mut Builder<'_>,
    scaled_time: NodeId,
    distance: NodeId,
    complement: NodeId,
) -> NodeId {
    let cut = b.lit_real(0.125);
    let short = b.call("le", &[scaled_time, cut]);
    let short_time = b.call("ifelse", &[short, scaled_time, cut]);
    let long_time = b.call("ifelse", &[short, cut, scaled_time]);

    let image = log_image_weight(b, short_time, distance, complement);
    let log_distance = b.call("log", &[distance]);
    let short_ratio = b.call("sub", &[image, log_distance]);

    let spectral = log_spectral_density(b, long_time, distance, complement);
    let log_time = b.call("log", &[long_time]);
    let root_time = b.call("sqrt", &[long_time]);
    let standardized = b.call("divide", &[distance, root_time]);
    let square = b.call("mul", &[standardized, standardized]);
    let half = b.lit_real(0.5);
    let half_square = b.call("mul", &[half, square]);
    let three_halves = b.lit_real(1.5);
    let power = b.call("mul", &[three_halves, log_time]);
    let sqrt_two_pi = b.lit_real(0.5 * TAU.ln());
    let inverse_log_shape = b.call("add", &[power, sqrt_two_pi]);
    let inverse_log_shape = b.call("add", &[inverse_log_shape, half_square]);
    let long_ratio = b.call("sub", &[spectral, log_distance]);
    let long_ratio = b.call("add", &[long_ratio, inverse_log_shape]);
    b.call("ifelse", &[short, short_ratio, long_ratio])
}

/// Method of images for zero drift, with the closest Gaussian exponent
/// removed. Pair opposite-signed images before summing, and factor the
/// distance to the nearer endpoint to preserve a small remaining density.
fn log_image_weight(
    b: &mut Builder<'_>,
    time: NodeId,
    distance: NodeId,
    complement: NodeId,
) -> NodeId {
    let half = b.lit_real(0.5);
    let one = b.lit_real(1.0);
    let two = b.lit_real(2.0);
    let minus_two = b.lit_real(-2.0);
    let near_hit = b.call("le", &[distance, half]);
    let near = b.call("ifelse", &[near_hit, distance, half]);
    let far = b.call("ifelse", &[near_hit, half, distance]);
    let gap = b.call("ifelse", &[near_hit, half, complement]);

    let mut near_sum = one;
    for k in 1..=2 {
        let k_lit = b.lit_real(f64::from(k));
        let two_k = b.lit_real(f64::from(2 * k));
        let minus_two_k = b.lit_real(f64::from(-2 * k));
        let minus_four_k = b.lit_real(f64::from(-4 * k));
        let difference = b.call("sub", &[k_lit, near]);
        let exponent = b.call("mul", &[minus_two_k, difference]);
        let exponent = b.call("divide", &[exponent, time]);
        let weight = b.call("exp", &[exponent]);
        let slope = b.call("divide", &[minus_four_k, time]);
        let change = scaled_expm1(b, slope, near);
        let reflected_distance = b.call("add", &[two_k, near]);
        let difference = b.call("mul", &[reflected_distance, change]);
        let pair = b.call("add", &[two, difference]);
        let term = b.call("mul", &[weight, pair]);
        near_sum = b.call("add", &[near_sum, term]);
    }

    let mut far_sum = b.lit_real(0.0);
    for k in 0..=2 {
        let weight = if k == 0 {
            one
        } else {
            let k_lit = b.lit_real(f64::from(k));
            let minus_two_k = b.lit_real(f64::from(-2 * k));
            let sum = b.call("add", &[k_lit, far]);
            let exponent = b.call("mul", &[minus_two_k, sum]);
            let exponent = b.call("divide", &[exponent, time]);
            b.call("exp", &[exponent])
        };
        let minus_image_spacing = b.lit_real(f64::from(-2 * (2 * k + 1)));
        let twice_index = b.lit_real(f64::from(2 * k + 2));
        let slope = b.call("divide", &[minus_image_spacing, time]);
        let change = scaled_expm1(b, slope, gap);
        let reflected_distance = b.call("sub", &[twice_index, far]);
        let difference = b.call("mul", &[reflected_distance, change]);
        let pair = b.call("sub", &[minus_two, difference]);
        let term = b.call("mul", &[weight, pair]);
        far_sum = b.call("add", &[far_sum, term]);
    }

    let normalized = b.call("ifelse", &[near_hit, near_sum, far_sum]);
    let interval_distance = b.call("ifelse", &[near_hit, near, gap]);
    let log_density = b.call("log", &[normalized]);
    let log_distance = b.call("log", &[interval_distance]);
    b.call("add", &[log_density, log_distance])
}

/// expm1(slope * distance) / distance, with a smooth limit for small arguments.
fn scaled_expm1(b: &mut Builder<'_>, slope: NodeId, distance: NodeId) -> NodeId {
    let argument = b.call("mul", &[slope, distance]);
    let boundary = b.lit_real(-1e-4);
    let use_series = b.call("ge", &[argument, boundary]);
    let safe_argument = b.call("ifelse", &[use_series, boundary, argument]);
    let numerator = b.call("expm1", &[safe_argument]);
    let one = b.lit_real(1.0);
    let safe_distance = b.call("ifelse", &[use_series, one, distance]);
    let quotient = b.call("divide", &[numerator, safe_distance]);

    let argument = b.call("ifelse", &[use_series, argument, boundary]);
    let half = b.lit_real(0.5);
    let sixth = b.lit_real(1.0 / 6.0);
    let twenty_fourth = b.lit_real(1.0 / 24.0);
    let series = b.call("mul", &[argument, twenty_fourth]);
    let series = b.call("add", &[sixth, series]);
    let series = b.call("mul", &[argument, series]);
    let series = b.call("add", &[half, series]);
    let series = b.call("mul", &[argument, series]);
    let series = b.call("add", &[one, series]);
    let zero = b.lit_real(0.0);
    let safe_slope = b.call("ifelse", &[use_series, slope, zero]);
    let series = b.call("mul", &[safe_slope, series]);
    b.call("ifelse", &[use_series, series, quotient])
}

/// Eigenfunction expansion for zero drift, normalized by the first mode.
/// sin(k*x) / sin(x) is the Chebyshev polynomial U[k-1](cos(x)).
fn log_spectral_density(
    b: &mut Builder<'_>,
    time: NodeId,
    distance: NodeId,
    complement: NodeId,
) -> NodeId {
    let half = b.lit_real(0.5);
    let near_hit = b.call("le", &[distance, half]);
    let near = b.call("ifelse", &[near_hit, distance, complement]);
    let pi = b.lit_real(PI);
    let angle = b.call("mul", &[pi, near]);
    let sine = b.call("sin", &[angle]);
    let cosine = b.call("cos", &[angle]);
    let negative = b.call("neg", &[cosine]);
    let cosine = b.call("ifelse", &[near_hit, cosine, negative]);
    let two = b.lit_real(2.0);
    let twice_cosine = b.call("mul", &[two, cosine]);
    let decay = b.lit_real(0.5 * PI * PI);
    let decay = b.call("mul", &[decay, time]);
    let one = b.lit_real(1.0);
    let mut sum = one;
    let mut previous = one;
    let mut current = twice_cosine;
    for k in 2..=8 {
        let multiplier = b.lit_real(f64::from(1 - k * k));
        let exponent = b.call("mul", &[multiplier, decay]);
        let weight = b.call("exp", &[exponent]);
        let k_lit = b.lit_real(f64::from(k));
        let term = b.call("mul", &[weight, k_lit]);
        let term = b.call("mul", &[term, current]);
        sum = b.call("add", &[sum, term]);
        if k < 8 {
            let next = b.call("mul", &[twice_cosine, current]);
            let next = b.call("sub", &[next, previous]);
            previous = current;
            current = next;
        }
    }
    let amplitude = b.call("mul", &[sine, sum]);
    let amplitude = b.call("mul", &[pi, amplitude]);
    let log_amplitude = b.call("log", &[amplitude]);
    b.call("sub", &[log_amplitude, decay])
}
