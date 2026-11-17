//! Piecewise polynomial interpolation on arbitrary, strictly increasing knots.

use super::*;

pub(super) fn lower(m: &mut Module, args: &[NodeId], member: &str) -> Option<(NodeId, NodeId)> {
    let (x, knots, values, policy) = match args {
        [x, knots, values] => (*x, *knots, *values, "constant".into()),
        [x, knots, values, policy] => {
            let policy = match m.node(resolve_ref(m, *policy).0) {
                Node::Lit(Scalar::Str(s)) => s.clone(),
                _ => return None,
            };
            (*x, *knots, *values, policy)
        }
        _ => return None,
    };
    if !matches!(policy.as_ref(), "constant" | "linear" | "error") {
        return None;
    }
    let xs = entries(m, knots)?;
    let ys = entries(m, values)?;
    let n = xs.len();
    if n < 2 || ys.len() != n {
        return None;
    }
    let zero = lit(m, 0.0);
    let one = lit(m, 1.0);
    let two = lit(m, 2.0);
    let three = lit(m, 3.0);
    let mut valid = build_call(m, "isfinite", &[x]);
    for &value in xs.iter().chain(&ys) {
        let finite = build_call(m, "isfinite", &[value]);
        valid = build_call(m, "land", &[valid, finite]);
    }
    let mut h = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let width = sub(m, xs[i + 1], xs[i]);
        let positive = build_call(m, "gt", &[width, zero]);
        valid = build_call(m, "land", &[valid, positive]);
        let safe_width = choose(m, positive, width, one);
        h.push(safe_width);
        let difference = sub(m, ys[i + 1], ys[i]);
        delta.push(divide(m, difference, safe_width));
    }
    let slopes = match member {
        "interp_linear" => None,
        "interp_cubic" => Some(natural_slopes(m, &h, &delta)),
        "interp_pchip" => Some(pchip_slopes(m, &h, &delta)),
        _ => return None,
    };
    let mut result = zero;
    for i in 0..n - 1 {
        let distance = sub(m, x, xs[i]);
        let t = divide(m, distance, h[i]);
        let value = if let Some(slopes) = &slopes {
            // Hermite form with coefficients in the normalized interval coordinate.
            let slope_sum = add(m, slopes[i], slopes[i + 1]);
            let twice_delta = mul(m, two, delta[i]);
            let cubic = sub(m, slope_sum, twice_delta);
            let three_delta = mul(m, three, delta[i]);
            let twice_left = mul(m, two, slopes[i]);
            let quadratic = sub(m, three_delta, twice_left);
            let quadratic = sub(m, quadratic, slopes[i + 1]);
            let value = mul(m, t, cubic);
            let value = add(m, quadratic, value);
            let value = mul(m, t, value);
            let value = add(m, slopes[i], value);
            let value = mul(m, t, value);
            let value = mul(m, h[i], value);
            add(m, ys[i], value)
        } else {
            let value = mul(m, distance, delta[i]);
            add(m, ys[i], value)
        };
        let lower = build_call(m, "ge", &[x, xs[i]]);
        result = choose(m, lower, value, result);
    }
    let below = build_call(m, "lt", &[x, xs[0]]);
    let above = build_call(m, "gt", &[x, xs[n - 1]]);
    let nan = lit(m, f64::NAN);
    let (left, right) = match policy.as_ref() {
        "constant" => (ys[0], ys[n - 1]),
        "linear" => {
            let left_slope = slopes.as_ref().map_or(delta[0], |s| s[0]);
            let right_slope = slopes.as_ref().map_or(delta[n - 2], |s| s[n - 1]);
            let left = sub(m, x, xs[0]);
            let left = mul(m, left, left_slope);
            let left = add(m, ys[0], left);
            let right = sub(m, x, xs[n - 1]);
            let right = mul(m, right, right_slope);
            (left, add(m, ys[n - 1], right))
        }
        _ => (nan, nan),
    };
    result = choose(m, below, left, result);
    result = choose(m, above, right, result);
    if policy.as_ref() == "error" {
        let exterior = build_call(m, "lor", &[below, above]);
        let interior = build_call(m, "lnot", &[exterior]);
        valid = build_call(m, "land", &[valid, interior]);
    }
    Some((choose(m, valid, result, nan), valid))
}

fn natural_slopes(m: &mut Module, h: &[NodeId], delta: &[NodeId]) -> Vec<NodeId> {
    let n = h.len() + 1;
    if n == 2 {
        return vec![delta[0]; 2];
    }
    let zero = lit(m, 0.0);
    let two = lit(m, 2.0);
    let six = lit(m, 6.0);
    let mut upper = vec![zero; n];
    let mut rhs = vec![zero; n];
    // Thomas elimination for the natural boundary second derivatives.
    for i in 1..n - 1 {
        let sum = add(m, h[i - 1], h[i]);
        let diagonal = mul(m, two, sum);
        let eliminated = mul(m, h[i - 1], upper[i - 1]);
        let diagonal = sub(m, diagonal, eliminated);
        upper[i] = divide(m, h[i], diagonal);
        let difference = sub(m, delta[i], delta[i - 1]);
        let target = mul(m, six, difference);
        let eliminated = mul(m, h[i - 1], rhs[i - 1]);
        let target = sub(m, target, eliminated);
        rhs[i] = divide(m, target, diagonal);
    }
    let mut second = vec![zero; n];
    for i in (1..n - 1).rev() {
        let eliminated = mul(m, upper[i], second[i + 1]);
        second[i] = sub(m, rhs[i], eliminated);
    }
    let mut slopes = Vec::with_capacity(n);
    for i in 0..n - 1 {
        let twice = mul(m, two, second[i]);
        let sum = add(m, twice, second[i + 1]);
        let correction = mul(m, h[i], sum);
        let correction = divide(m, correction, six);
        slopes.push(sub(m, delta[i], correction));
    }
    let twice = mul(m, two, second[n - 1]);
    let sum = add(m, second[n - 2], twice);
    let correction = mul(m, h[n - 2], sum);
    let correction = divide(m, correction, six);
    slopes.push(add(m, delta[n - 2], correction));
    slopes
}

fn pchip_slopes(m: &mut Module, h: &[NodeId], delta: &[NodeId]) -> Vec<NodeId> {
    if h.len() == 1 {
        return vec![delta[0]; 2];
    }
    let zero = lit(m, 0.0);
    let one = lit(m, 1.0);
    let two = lit(m, 2.0);
    let mut slopes = Vec::with_capacity(h.len() + 1);
    slopes.push(endpoint(m, h[0], h[1], delta[0], delta[1]));
    for i in 1..h.len() {
        let same = same_nonzero_sign(m, delta[i - 1], delta[i]);
        let left = choose(m, same, delta[i - 1], one);
        let right = choose(m, same, delta[i], one);
        let twice = mul(m, two, h[i]);
        let w1 = add(m, twice, h[i - 1]);
        let twice = mul(m, two, h[i - 1]);
        let w2 = add(m, h[i], twice);
        let numerator = add(m, w1, w2);
        let a = divide(m, w1, left);
        let b = divide(m, w2, right);
        let denominator = add(m, a, b);
        let slope = divide(m, numerator, denominator);
        slopes.push(choose(m, same, slope, zero));
    }
    let last = h.len() - 1;
    slopes.push(endpoint(
        m,
        h[last],
        h[last - 1],
        delta[last],
        delta[last - 1],
    ));
    slopes
}

fn endpoint(m: &mut Module, h0: NodeId, h1: NodeId, d0: NodeId, d1: NodeId) -> NodeId {
    let zero = lit(m, 0.0);
    let two = lit(m, 2.0);
    let three = lit(m, 3.0);
    let twice = mul(m, two, h0);
    let weight = add(m, twice, h1);
    let a = mul(m, weight, d0);
    let b = mul(m, h0, d1);
    let numerator = sub(m, a, b);
    let denominator = add(m, h0, h1);
    let slope = divide(m, numerator, denominator);
    let correct_sign = same_nonzero_sign(m, slope, d0);
    let wrong_sign = build_call(m, "lnot", &[correct_sign]);
    let opposite = same_nonzero_sign(m, d0, d1);
    let turns = build_call(m, "lnot", &[opposite]);
    let bound = mul(m, three, d0);
    let abs_slope = build_call(m, "abs", &[slope]);
    let abs_bound = build_call(m, "abs", &[bound]);
    let large = build_call(m, "gt", &[abs_slope, abs_bound]);
    let limit = build_call(m, "land", &[turns, large]);
    let slope = choose(m, limit, bound, slope);
    choose(m, wrong_sign, zero, slope)
}

fn same_nonzero_sign(m: &mut Module, a: NodeId, b: NodeId) -> NodeId {
    let zero = lit(m, 0.0);
    let ap = build_call(m, "gt", &[a, zero]);
    let bp = build_call(m, "gt", &[b, zero]);
    let positive = build_call(m, "land", &[ap, bp]);
    let an = build_call(m, "lt", &[a, zero]);
    let bn = build_call(m, "lt", &[b, zero]);
    let negative = build_call(m, "land", &[an, bn]);
    build_call(m, "lor", &[positive, negative])
}

pub(super) fn entries(m: &mut Module, value: NodeId) -> Option<Vec<NodeId>> {
    let resolved = resolve_ref(m, value).0;
    let count = match m.type_of(value).or_else(|| m.type_of(resolved)) {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => match shape[0] {
            flatppl_core::Dim::Static(n) => n as usize,
            _ => return None,
        },
        _ => return None,
    };
    Some(
        (1..=count)
            .map(|i| {
                let index = m.alloc(Node::Lit(Scalar::Int(i as i64)));
                build_call(m, "get", &[value, index])
            })
            .collect(),
    )
}

pub(super) fn choose(m: &mut Module, condition: NodeId, yes: NodeId, no: NodeId) -> NodeId {
    build_call(m, "ifelse", &[condition, yes, no])
}
