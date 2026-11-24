use flatppl_core::graph::build_call;
use flatppl_core::{INTEGRATION_POINT, LOG_INTEGRAL, Module, NodeId};

use crate::{Emitter, Step, callbacks, real, rules::special_call, tensor};

pub(super) fn contribution(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    second: bool,
) -> Option<NodeId> {
    let [a, b, x] = *step.args.as_slice() else {
        return None;
    };
    let operands = [a, b, x].map(|arg| emitter.primal(arg));
    let log_mass = step
        .op
        .starts_with("log_")
        .then(|| emitter.primal(step.node));
    let pointwise = step.pointwise(&emitter.module);
    let module = &mut emitter.module;
    if !pointwise {
        return Some(scalar_contribution(
            module, step.op, operands, bar, log_mass, second,
        ));
    }

    let mut entries = Vec::new();
    let local = ["beta_shape_a", "beta_shape_b", "beta_bound"]
        .map(|name| callbacks::callback_input(module, name, &mut entries));
    let seed = callbacks::callback_input(module, "beta_seed", &mut entries);
    let mass = log_mass.map(|_| callbacks::callback_input(module, "beta_log_mass", &mut entries));
    let body = scalar_contribution(module, step.op, local, seed, mass, second);
    let function = callbacks::reify(module, body, entries);
    let mut args = vec![function, operands[0], operands[1], operands[2], bar];
    args.extend(log_mass);
    Some(build_call(module, "broadcast", &args))
}

fn scalar_contribution(
    module: &mut Module,
    op: &str,
    [a, b, x]: [NodeId; 3],
    bar: NodeId,
    log_mass: Option<NodeId>,
    second: bool,
) -> NodeId {
    let zero = real(module, 0.0);
    let half = real(module, 0.5);
    let one = real(module, 1.0);
    let inf = real(module, f64::INFINITY);
    let nan = real(module, f64::NAN);
    let token = tensor::int(module, module.node_count() as i64);
    let u = build_call(module, INTEGRATION_POINT, &[token]);
    let mut call = |op: &str, args: &[NodeId]| build_call(module, op, args);

    let sum = call("add", &[a, b]);
    let finite = call("isfinite", &[sum]);
    let positive_a = call("gt", &[a, zero]);
    let positive_b = call("gt", &[b, zero]);
    let valid = call("land", &[positive_a, positive_b]);
    let valid = call("land", &[finite, valid]);
    let in_left = call("ge", &[x, zero]);
    let in_right = call("le", &[x, one]);
    let in_support = call("land", &[in_left, in_right]);
    let valid = call("land", &[valid, in_support]);

    let log_x = call("log", &[x]);
    let negative_x = call("neg", &[x]);
    let log_complement = call("log1p", &[negative_x]);
    let left_endpoint = call("isinf", &[log_x]);
    let right_endpoint = call("isinf", &[log_complement]);
    let endpoint = call("lor", &[left_endpoint, right_endpoint]);
    let interior = call("lnot", &[endpoint]);
    let safe = call("land", &[valid, interior]);
    // Internal tail evaluations are checked even for an unchecked primal.
    let a = call("ifelse", &[safe, a, one]);
    let b = call("ifelse", &[safe, b, one]);
    let x = call("ifelse", &[safe, x, half]);
    let complement = call("sub", &[one, x]);
    let [shape, other, bound, remainder] = if second {
        [b, a, complement, x]
    } else {
        [a, b, x, complement]
    };
    let shape_bound_log = if second { log_complement } else { log_x };
    let log_half = real(module, -std::f64::consts::LN_2);
    let psi = special_call(module, "digamma", vec![shape], false);
    let sum = build_call(module, "add", &[shape, other]);
    let psi_sum = special_call(module, "digamma", vec![sum], false);
    let mut call = |op: &str, args: &[NodeId]| build_call(module, op, args);

    // The beta shape score has mean zero. Pick the side where its sign
    // is fixed throughout, reflecting the upper side onto a lower interval.
    let log_bound = call("ifelse", &[safe, shape_bound_log, log_half]);
    let center = call("sub", &[psi, psi_sum]);
    let score = call("sub", &[log_bound, center]);
    let lower = call("le", &[score, zero]);
    let limit = call("ifelse", &[lower, bound, remainder]);
    let gap = call("ifelse", &[lower, remainder, bound]);
    let alpha = call("ifelse", &[lower, shape, other]);
    let beta = call("ifelse", &[lower, other, shape]);
    let beta_minus_one = call("sub", &[beta, one]);

    // t = limit * exp(-u / rate). The Jacobian removes the endpoint
    // power singularity. Constant factors cancel in the conditional ratio.
    let ratio = call("divide", &[limit, gap]);
    let linear = call("mul", &[beta_minus_one, ratio]);
    let linear = call("sub", &[alpha, linear]);
    let root_alpha = call("sqrt", &[alpha]);
    let rate = call("max", &[linear, root_alpha]);
    let rate = call("min", &[rate, alpha]);
    let v = call("divide", &[u, rate]);
    let v = call("neg", &[v]);
    let change = call("expm1", &[v]);
    let change = call("neg", &[change]);
    let delta = call("mul", &[limit, change]);
    let relative_change = call("divide", &[delta, gap]);
    let finite = call("isfinite", &[relative_change]);
    let complement_change = call("log1p", &[relative_change]);
    let moved_gap = call("add", &[gap, delta]);
    let log_moved_gap = call("log", &[moved_gap]);
    let log_gap = call("log", &[gap]);
    let alternative = call("sub", &[log_moved_gap, log_gap]);
    let complement_change = call("ifelse", &[finite, complement_change, alternative]);
    let power = call("mul", &[alpha, v]);
    let complement_power = call("mul", &[beta_minus_one, complement_change]);
    let kernel = call("add", &[power, complement_power]);
    let lower_magnitude = call("add", &[score, v]);
    let lower_magnitude = call("neg", &[lower_magnitude]);
    let upper_magnitude = call("add", &[score, complement_change]);
    let magnitude = call("ifelse", &[lower, lower_magnitude, upper_magnitude]);
    let magnitude = call("log", &[magnitude]);
    let weighted = call("add", &[kernel, magnitude]);
    let weighted = call(LOG_INTEGRAL, &[weighted, u, zero, inf]);
    let mass = call(LOG_INTEGRAL, &[kernel, u, zero, inf]);
    let log_expectation = call("sub", &[weighted, mass]);

    let selected_lower = if second {
        call("lnot", &[lower])
    } else {
        lower
    };
    let requested_upper = op.ends_with("cc");
    let tail = |module: &mut Module, upper| {
        special_call(
            module,
            if upper { "log_betaincc" } else { "log_betainc" },
            vec![a, b, x],
            false,
        )
    };
    let log_scale = if let Some(log_mass) = log_mass {
        let opposite = tail(module, !requested_upper);
        let difference = build_call(module, "sub", &[opposite, log_mass]);
        if requested_upper {
            build_call(module, "ifelse", &[selected_lower, difference, zero])
        } else {
            build_call(module, "ifelse", &[selected_lower, zero, difference])
        }
    } else {
        let log_p = tail(module, false);
        let log_q = tail(module, true);
        build_call(module, "ifelse", &[selected_lower, log_p, log_q])
    };
    let log_derivative = build_call(module, "add", &[log_expectation, log_scale]);
    let slope = build_call(module, "exp", &[log_derivative]);
    let slope = if requested_upper == second {
        build_call(module, "neg", &[slope])
    } else {
        slope
    };
    let contribution = build_call(module, "mul", &[bar, slope]);
    let contribution = build_call(module, "ifelse", &[endpoint, zero, contribution]);
    build_call(module, "ifelse", &[valid, contribution, nan])
}
