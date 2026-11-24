use flatppl_core::graph::build_call;
use flatppl_core::{Module, NodeId};

use crate::{real, rules::special_call, tensor};

// DLMF 10.15.1. For 0 <= ν <= 16 and 1/8 <= x <= 4 the first
// omitted derivative term is bounded by 2^16 (4^32/(32!)^2)
// (log(16) + log(49)). Subsequent bounds have ratio below 0.004.
pub(super) fn bessel_order(
    module: &mut Module,
    order: NodeId,
    argument: NodeId,
    kind: &str,
) -> NodeId {
    use crate::callbacks::{callback_input, reify};

    let mut parameters = Vec::new();
    let k = callback_input(module, "term", &mut parameters);
    let nu = callback_input(module, "order", &mut parameters);
    let log_half = callback_input(module, "loghalf", &mut parameters);
    let zero = tensor::int(module, 0);
    let one = tensor::int(module, 1);
    let two = tensor::int(module, 2);
    let one_real = real(module, 1.0);
    let minus_one = real(module, -1.0);
    let parity = build_call(module, "mod", &[k, two]);
    let even = build_call(module, "equal", &[parity, zero]);
    let sign = build_call(module, "ifelse", &[even, one_real, minus_one]);
    let kp1 = build_call(module, "add", &[k, one]);
    let gamma_arg = build_call(module, "add", &[nu, kp1]);
    let log_k_factorial = build_call(module, "loggamma", &[kp1]);
    let log_gamma = build_call(module, "loggamma", &[gamma_arg]);
    let twice_k = build_call(module, "mul", &[two, k]);
    let power = build_call(module, "add", &[nu, twice_k]);
    let log_power = build_call(module, "mul", &[power, log_half]);
    let exponent = build_call(module, "sub", &[log_power, log_k_factorial]);
    let exponent = build_call(module, "sub", &[exponent, log_gamma]);
    let magnitude = build_call(module, "exp", &[exponent]);
    let coefficient = if kind == "j" {
        build_call(module, "mul", &[sign, magnitude])
    } else {
        magnitude
    };
    let digamma = special_call(module, "digamma", vec![gamma_arg], false);
    let factor = build_call(module, "sub", &[log_half, digamma]);
    let term = build_call(module, "mul", &[coefficient, factor]);
    let function = reify(module, term, parameters);

    let terms = tensor::int(module, 32);
    let zeros = build_call(module, "zeros", &[terms]);
    let indices = build_call(module, "indicesof0", &[zeros]);
    let abs_x = build_call(module, "abs", &[argument]);
    let log_x = build_call(module, "log", &[abs_x]);
    let log_two = real(module, std::f64::consts::LN_2);
    let log_half = build_call(module, "sub", &[log_x, log_two]);
    let held_order = build_call(module, "vector", &[order]);
    let held_log_half = build_call(module, "vector", &[log_half]);
    let terms = build_call(
        module,
        "broadcast",
        &[function, indices, held_order, held_log_half],
    );
    let slope = build_call(module, "sum", &[terms]);
    if kind == "i-log" {
        let value = special_call(module, "log_bessel_i", vec![order, abs_x], false);
        let value = build_call(module, "exp", &[value]);
        return build_call(module, "divide", &[slope, value]);
    }
    let scaled = kind.ends_with("-scaled");
    let slope = if scaled {
        let negative = build_call(module, "neg", &[abs_x]);
        let factor = build_call(module, "exp", &[negative]);
        build_call(module, "mul", &[factor, slope])
    } else {
        slope
    };
    let function = if kind == "j" {
        "bessel_j"
    } else if scaled {
        "bessel_i_scaled"
    } else {
        "bessel_i"
    };
    let base = special_call(module, function, vec![order, abs_x], false);
    let base = build_call(module, "real", &[base]);
    let zero = real(module, 0.0);
    let negative = build_call(module, "lt", &[argument, zero]);
    let pi = real(module, std::f64::consts::PI);
    let phase = build_call(module, "mul", &[pi, order]);
    let sin = build_call(module, "sin", &[phase]);
    let cos = build_call(module, "cos", &[phase]);
    let base = build_call(module, "mul", &[pi, base]);
    let re_slope = build_call(module, "mul", &[slope, cos]);
    let re_phase = build_call(module, "mul", &[base, sin]);
    let re = build_call(module, "sub", &[re_slope, re_phase]);
    let re = build_call(module, "ifelse", &[negative, re, slope]);
    let im_slope = build_call(module, "mul", &[slope, sin]);
    let im_phase = build_call(module, "mul", &[base, cos]);
    let im = build_call(module, "add", &[im_slope, im_phase]);
    let im = build_call(module, "ifelse", &[negative, im, zero]);
    build_call(module, "complex", &[re, im])
}

pub(super) fn bessel_k_order(
    module: &mut Module,
    order: NodeId,
    z: NodeId,
    member: &str,
) -> NodeId {
    use flatppl_core::{INTEGRATION_POINT, LOG_INTEGRAL};

    let zero = real(module, 0.0);
    let one = real(module, 1.0);
    let half = real(module, 0.5);
    let neg_two = real(module, -2.0);
    let log_half = real(module, -std::f64::consts::LN_2);
    let inf = real(module, f64::INFINITY);
    let token = tensor::int(module, module.node_count() as i64);
    let u = build_call(module, INTEGRATION_POINT, &[token]);
    let log_k = if member == "log_bessel_k" {
        Some(special_call(module, "log_bessel_k", vec![order, z], false))
    } else {
        None
    };
    let mut call = |op: &str, args: &[NodeId]| build_call(module, op, args);
    let zero_order = call("iszero", &[order]);
    let abs_order = call("abs", &[order]);
    let abs_order = call("ifelse", &[zero_order, one, abs_order]);
    // DLMF 10.32.9, differentiate cosh(v*t), then use u = exp(t)-1.
    // cosh(t)-1 = u*u/(2*(1+u)); the scaled exponent avoids large exp terms.
    let t = call("log1p", &[u]);
    let y = call("add", &[one, u]);
    let ratio = call("divide", &[u, y]);
    let cosh_minus_one = call("mul", &[u, ratio]);
    let cosh_minus_one = call("mul", &[half, cosh_minus_one]);
    let decay = call("mul", &[z, cosh_minus_one]);
    let phase = call("mul", &[abs_order, t]);
    let sinh_correction = call("mul", &[neg_two, phase]);
    let sinh_correction = call("expm1", &[sinh_correction]);
    let sinh_correction = call("neg", &[sinh_correction]);
    let sinh_correction = call("log", &[sinh_correction]);
    let log_t = call("log", &[t]);
    let weight = call("sub", &[phase, decay]);
    let weight = call("add", &[weight, sinh_correction]);
    let weight = call("add", &[weight, log_t]);
    let weight = call("add", &[weight, log_half]);
    let weight = call("sub", &[weight, t]);
    let log_slope = call(LOG_INTEGRAL, &[weight, u, zero, inf]);
    let log_slope = if let Some(log_k) = log_k {
        let scale = call("add", &[log_k, z]);
        call("sub", &[log_slope, scale])
    } else if member == "bessel_k" {
        call("sub", &[log_slope, z])
    } else {
        log_slope
    };
    let slope = call("exp", &[log_slope]);
    let negative = call("lt", &[order, zero]);
    let minus = call("neg", &[slope]);
    let signed = call("ifelse", &[negative, minus, slope]);
    call("ifelse", &[zero_order, zero, signed])
}

pub(super) fn bessel_y_order(module: &mut Module, order: NodeId, z: NodeId) -> NodeId {
    use flatppl_core::{INTEGRATION_POINT, LOG_INTEGRAL, MASS_INTEGRAL};

    let zero = real(module, 0.0);
    let one = real(module, 1.0);
    let half = real(module, 0.5);
    let pi = real(module, std::f64::consts::PI);
    let angular_shift = real(module, std::f64::consts::PI.powi(2) / 2.0);
    let inf = real(module, f64::INFINITY);
    let token = tensor::int(module, module.node_count() as i64);
    let u = build_call(module, INTEGRATION_POINT, &[token]);
    let mut call = |op: &str, args: &[NodeId]| build_call(module, op, args);
    // Differentiate DLMF 10.9.7. Shift the cosine to integrate a positive
    // angular weight, then subtract its exact constant contribution.
    let sin_u = call("sin", &[u]);
    let angle = call("mul", &[z, sin_u]);
    let order_u = call("mul", &[order, u]);
    let angle = call("sub", &[angle, order_u]);
    let cosine = call("cos", &[angle]);
    let shifted = call("add", &[one, cosine]);
    let weight = call("mul", &[u, shifted]);
    let angular = call(MASS_INTEGRAL, &[weight, u, zero, pi]);
    let angular = call("sub", &[angular, angular_shift]);

    // The improper terms remain positive when integrated separately.
    // With t = log(1+u), sinh(t) = (u + u/(1+u))/2.
    let y = call("add", &[one, u]);
    let t = call("log1p", &[u]);
    let ratio = call("divide", &[u, y]);
    let sinh = call("add", &[u, ratio]);
    let sinh = call("mul", &[half, sinh]);
    let decay = call("mul", &[z, sinh]);
    let decay = call("add", &[t, decay]);
    let phase = call("mul", &[order, t]);
    let minus = call("add", &[phase, decay]);
    let minus = call("neg", &[minus]);
    let plus = call("sub", &[phase, decay]);
    let log_t = call("log", &[t]);
    let weighted_plus = call("add", &[plus, log_t]);
    let weighted_minus = call("add", &[minus, log_t]);
    let plus = call(LOG_INTEGRAL, &[weighted_plus, u, zero, inf]);
    let minus_t = call(LOG_INTEGRAL, &[weighted_minus, u, zero, inf]);
    let minus_plain = call(LOG_INTEGRAL, &[minus, u, zero, inf]);
    let plus = call("exp", &[plus]);
    let minus_t = call("exp", &[minus_t]);
    let minus_plain = call("exp", &[minus_plain]);
    let pi_order = call("mul", &[pi, order]);
    let cos = call("cos", &[pi_order]);
    let sin = call("sin", &[pi_order]);
    let t_part = call("mul", &[cos, minus_t]);
    let endpoint = call("mul", &[sin, minus_plain]);
    let first = call("add", &[angular, plus]);
    let first = call("sub", &[t_part, first]);
    let first = call("divide", &[first, pi]);
    call("add", &[first, endpoint])
}

pub(super) fn airy_slope(module: &mut Module, x: NodeId) -> NodeId {
    // DLMF 9.6.3–6. Near zero, Ai'' = x Ai gives the Taylor coefficients.
    let zero = real(module, 0.0);
    let two_thirds = real(module, 2.0 / 3.0);
    let minus_order = real(module, -2.0 / 3.0);
    let small = real(module, 1e-3);
    let near = real(module, 0.5);
    let abs_x = build_call(module, "abs", &[x]);
    let small_x = build_call(module, "lt", &[abs_x, small]);
    let abs_safe = build_call(module, "ifelse", &[small_x, near, abs_x]);
    let root = build_call(module, "sqrt", &[abs_safe]);
    let z = build_call(module, "mul", &[abs_safe, root]);
    let z = build_call(module, "mul", &[two_thirds, z]);
    let finite = build_call(module, "isfinite", &[z]);
    let z = build_call(module, "ifelse", &[finite, z, near]);
    let left = special_call(module, "bessel_j", vec![minus_order, z], false);
    let right = special_call(module, "bessel_j", vec![two_thirds, z], false);
    let left = build_call(module, "real", &[left]);
    let right = build_call(module, "real", &[right]);
    let difference = build_call(module, "sub", &[left, right]);
    let third = real(module, 1.0 / 3.0);
    let scale = build_call(module, "mul", &[third, x]);
    let negative = build_call(module, "mul", &[scale, difference]);
    let log_k = special_call(module, "log_bessel_k", vec![two_thirds, z], false);
    let log_x = build_call(module, "log", &[abs_safe]);
    let log_scale = real(module, (std::f64::consts::PI * 3.0_f64.sqrt()).ln());
    let log_magnitude = build_call(module, "add", &[log_x, log_k]);
    let log_magnitude = build_call(module, "sub", &[log_magnitude, log_scale]);
    let magnitude = build_call(module, "exp", &[log_magnitude]);
    let positive = build_call(module, "neg", &[magnitude]);
    let positive = build_call(module, "ifelse", &[finite, positive, zero]);
    let nan = real(module, f64::NAN);
    let negative = build_call(module, "ifelse", &[finite, negative, nan]);
    let is_negative = build_call(module, "lt", &[x, zero]);
    let main = build_call(module, "ifelse", &[is_negative, negative, positive]);

    let constant = real(module, -0.2588194037928068);
    let c2 = real(module, 0.3550280538878172 / 2.0);
    let c3 = real(module, -0.2588194037928068 / 3.0);
    let square = build_call(module, "mul", &[x, x]);
    let cubic_factor = build_call(module, "mul", &[x, c3]);
    let polynomial = build_call(module, "add", &[c2, cubic_factor]);
    let correction = build_call(module, "mul", &[square, polynomial]);
    let near_origin = build_call(module, "add", &[constant, correction]);
    let slope = build_call(module, "ifelse", &[small_x, near_origin, main]);
    let undefined = build_call(module, "isnan", &[x]);
    build_call(module, "ifelse", &[undefined, nan, slope])
}
