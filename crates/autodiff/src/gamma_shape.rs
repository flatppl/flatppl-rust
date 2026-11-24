use flatppl_core::graph::build_call;
use flatppl_core::{INTEGRATION_POINT, LOG_INTEGRAL, Module, NodeId};

use crate::{Emitter, Error, Step, callbacks, real, rules::special_call, tensor};

pub(super) fn contribution(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
) -> Result<NodeId, Error> {
    let [shape, argument] = step.args.as_slice() else {
        unreachable!("gamma shape rules have two operands");
    };
    let a = emitter.primal(*shape);
    let x = emitter.primal(*argument);
    let log_mass = match step.op {
        "log_gammainc" | "log_gammaincc" => Some(emitter.primal(step.node)),
        _ => None,
    };

    let pointwise = step.pointwise(&emitter.module);
    let module = &mut emitter.module;
    if !pointwise {
        return Ok(scalar_contribution(module, step.op, a, x, bar, log_mass));
    }

    let mut entries = Vec::new();
    let shape = callbacks::callback_input(module, "gamma_shape", &mut entries);
    let argument = callbacks::callback_input(module, "gamma_argument", &mut entries);
    let seed = callbacks::callback_input(module, "gamma_seed", &mut entries);
    let output =
        log_mass.map(|_| callbacks::callback_input(module, "gamma_log_mass", &mut entries));
    let body = scalar_contribution(module, step.op, shape, argument, seed, output);
    let function = callbacks::reify(module, body, entries);
    let mut arguments = vec![function, a, x, bar];
    arguments.extend(log_mass);
    Ok(build_call(module, "broadcast", &arguments))
}

fn scalar_contribution(
    module: &mut Module,
    op: &str,
    a: NodeId,
    x: NodeId,
    bar: NodeId,
    log_mass: Option<NodeId>,
) -> NodeId {
    let zero = real(module, 0.0);
    let one = real(module, 1.0);
    let infinity = real(module, f64::INFINITY);
    let nan = real(module, f64::NAN);
    let token = tensor::int(module, module.node_count() as i64);
    let u = build_call(module, INTEGRATION_POINT, &[token]);
    let nonnegative_x = build_call(module, "ge", &[x, zero]);
    let positive_a = build_call(module, "gt", &[a, zero]);
    let finite_a = build_call(module, "isfinite", &[a]);
    let valid_shape = build_call(module, "land", &[positive_a, finite_a]);
    let valid = build_call(module, "land", &[valid_shape, nonnegative_x]);
    // The helper log tails are checked, including when used for an unchecked
    // gammainc. Keep invalid inputs out of the helpers in both AD modes.
    let a = build_call(module, "ifelse", &[valid, a, one]);
    let x = build_call(module, "ifelse", &[valid, x, one]);
    let psi = special_call(module, "digamma", vec![a], false);
    let small_limit = real(module, 1.0 / 16.0);
    let exp_coefficients = (2..=10)
        .map(|n| real(module, 1.0 / f64::from((1..=n).product::<i32>())))
        .collect::<Vec<_>>();
    let exp_coefficients = build_call(module, "vector", &exp_coefficients);
    let log_coefficients = (2..=15)
        .map(|n| {
            let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
            real(module, sign / f64::from(n))
        })
        .collect::<Vec<_>>();
    let log_coefficients = build_call(module, "vector", &log_coefficients);
    let mut call = |op: &str, args: &[NodeId]| build_call(module, op, args);
    let remainder_series = |call: &mut dyn FnMut(&str, &[NodeId]) -> NodeId, z, coefficients| {
        let square = call("mul", &[z, z]);
        let polynomial = call("polynomial", &[coefficients, z]);
        call("mul", &[square, polynomial])
    };

    // Use the same logarithm that determines the gamma tail. A target that
    // flushes subnormal arguments can evaluate log(x) as -infinity.
    let log_x = call("log", &[x]);
    let endpoint = call("isinf", &[log_x]);
    let x = call("ifelse", &[endpoint, one, x]);
    let log_x = call("ifelse", &[endpoint, zero, log_x]);
    let score = call("sub", &[log_x, psi]);
    let lower = call("le", &[score, zero]);
    let root_a = call("sqrt", &[a]);

    // The gamma score is log(t) - digamma(a), with zero total integral.
    // Integrate the negative score below x or the positive score above x.
    // Choosing by the sign at x makes the integrand nonnegative everywhere.
    // Below x, t = x * exp(-u / rate) cancels the power singularity at t = 0.
    let gap = call("sub", &[a, x]);
    let rate = call("max", &[gap, root_a]);
    let rate = call("min", &[rate, a]);
    let v = call("divide", &[u, rate]);
    let lower_score = call("sub", &[v, score]);
    let lower_score = call("log", &[lower_score]);
    let log_rate = call("log", &[rate]);
    let v = call("neg", &[v]);
    let delta = call("expm1", &[v]);
    // Evaluate expm1(v) - v near zero with its Taylor remainder to working
    // precision. Recombine the bounded part and remainder using
    // expm1(b + r) = expm1(b) + expm1(r) + expm1(b) * expm1(r).
    let negative_limit = call("neg", &[small_limit]);
    let bounded = call("max", &[negative_limit, v]);
    let rest = call("sub", &[v, bounded]);
    let exp_bounded = call("expm1", &[bounded]);
    let exp_rest = call("expm1", &[rest]);
    let cross = call("mul", &[exp_bounded, exp_rest]);
    let stable = remainder_series(&mut call, bounded, exp_coefficients);
    let rest = call("sub", &[exp_rest, rest]);
    let remainder = call("add", &[stable, rest]);
    let remainder = call("add", &[remainder, cross]);
    let remainder = call("neg", &[remainder]);
    let remainder = call("mul", &[a, remainder]);
    let linear = call("mul", &[gap, delta]);
    let lower_kernel = call("add", &[remainder, linear]);
    let lower_kernel = call("sub", &[lower_kernel, log_rate]);

    // Above x, t = x + scale * u tracks the width near the gamma mode
    // and approaches a unit scale in the far upper tail.
    let gap = call("neg", &[gap]);
    let gap = call("max", &[gap, one]);
    let scale = call("divide", &[x, gap]);
    let scale = call("min", &[scale, root_a]);
    let scale = call("max", &[scale, one]);
    let delta = call("mul", &[scale, u]);
    let ratio = call("divide", &[delta, x]);
    let log_ratio = call("log1p", &[ratio]);
    let upper_score = call("add", &[score, log_ratio]);
    let upper_score = call("log", &[upper_score]);
    let shape_minus_one = call("sub", &[a, one]);
    // For log1p(z) - z, use the corresponding product decomposition
    // z = b + r + b*r, so log1p(z) = log1p(b) + log1p(r).
    let bounded = call("min", &[ratio, small_limit]);
    let rest = call("sub", &[ratio, bounded]);
    let one_plus = call("add", &[one, bounded]);
    let rest = call("divide", &[rest, one_plus]);
    let cross = call("mul", &[bounded, rest]);
    let stable = remainder_series(&mut call, bounded, log_coefficients);
    let log_rest = call("log1p", &[rest]);
    let rest = call("sub", &[log_rest, rest]);
    let remainder = call("add", &[stable, rest]);
    let remainder = call("sub", &[remainder, cross]);
    let remainder = call("mul", &[shape_minus_one, remainder]);
    let linear = call("sub", &[shape_minus_one, x]);
    let linear = call("mul", &[linear, ratio]);
    let log_scale = call("log", &[scale]);
    let jacobian = call("sub", &[log_scale, log_x]);
    let upper_kernel = call("add", &[remainder, linear]);
    let upper_kernel = call("add", &[upper_kernel, jacobian]);

    // For a small upper bound x, step in log(t) to resolve mass spread
    // across t close to zero. This includes the Jacobian t * scale.
    let small_x = call("lt", &[x, one]);
    let neg_log_x = call("neg", &[log_x]);
    let scale = call("max", &[neg_log_x, one]);
    let v = call("mul", &[scale, u]);
    let small_score = call("add", &[score, v]);
    let small_score = call("log", &[small_score]);
    let log_t = call("add", &[log_x, v]);
    let t = call("exp", &[log_t]);
    let delta = call("sub", &[t, x]);
    let power = call("mul", &[a, v]);
    let log_scale = call("log", &[scale]);
    let small_kernel = call("sub", &[power, delta]);
    let small_kernel = call("add", &[small_kernel, log_scale]);
    let upper_kernel = call("ifelse", &[small_x, small_kernel, upper_kernel]);
    let upper_score = call("ifelse", &[small_x, small_score, upper_score]);

    // Take the score expectation in the selected tail and multiply by its
    // probability. The ratio cancels the common gamma density prefactor
    // without subtracting large, nearly equal log normalizers.
    let kernel = call("ifelse", &[lower, lower_kernel, upper_kernel]);
    let log_score = call("ifelse", &[lower, lower_score, upper_score]);
    let weighted = call("add", &[kernel, log_score]);
    let weighted = call(LOG_INTEGRAL, &[weighted, u, zero, infinity]);
    let kernel_mass = call(LOG_INTEGRAL, &[kernel, u, zero, infinity]);
    let log_expectation = call("sub", &[weighted, kernel_mass]);

    let log_tail =
        |module: &mut flatppl_core::Module, name| special_call(module, name, vec![a, x], false);
    let difference = match (op, log_mass) {
        ("log_gammainc", Some(log_mass)) => {
            let opposite = log_tail(module, "log_gammaincc");
            let difference = build_call(module, "sub", &[opposite, log_mass]);
            build_call(module, "ifelse", &[lower, zero, difference])
        }
        ("log_gammaincc", Some(log_mass)) => {
            let opposite = log_tail(module, "log_gammainc");
            let difference = build_call(module, "sub", &[opposite, log_mass]);
            build_call(module, "ifelse", &[lower, difference, zero])
        }
        _ => {
            let log_p = log_tail(module, "log_gammainc");
            let log_q = log_tail(module, "log_gammaincc");
            build_call(module, "ifelse", &[lower, log_p, log_q])
        }
    };
    let mut call = |op: &str, args: &[NodeId]| build_call(module, op, args);
    let log_derivative = call("add", &[log_expectation, difference]);
    let derivative = call("exp", &[log_derivative]);
    let signed = match op {
        "gammainc" | "log_gammainc" => call("neg", &[derivative]),
        _ => derivative,
    };
    let weighted = call("mul", &[bar, signed]);
    // All four tails are constant in shape at a fixed endpoint.
    let contribution = call("ifelse", &[endpoint, zero, weighted]);
    call("ifelse", &[valid, contribution, nan])
}
