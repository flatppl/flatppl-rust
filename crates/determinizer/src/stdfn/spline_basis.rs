//! Cox–de Boor recurrences and exact span quadrature over the base interval.

use super::spline_interpolation::{choose, entries};
use super::*;

pub(super) fn lower(m: &mut Module, args: &[NodeId], member: &str) -> Option<(NodeId, NodeId)> {
    let (knots, degree_arg, order_arg) = match (member, args) {
        ("bspline_basis", [_, knots, degree]) => (*knots, *degree, None),
        ("bspline_derivative", [_, knots, degree]) => (*knots, *degree, None),
        ("bspline_derivative", [_, knots, degree, order]) => (*knots, *degree, Some(*order)),
        ("bspline_integral", [_, _, knots, degree]) => (*knots, *degree, None),
        ("bspline_penalty", [knots, degree]) => (*knots, *degree, None),
        ("bspline_penalty", [knots, degree, order]) => (*knots, *degree, Some(*order)),
        _ => return None,
    };
    let degree = usize::try_from(literal_int(m, degree_arg)?).ok()?;
    let default_order = match member {
        "bspline_derivative" => 1,
        "bspline_penalty" => 2,
        _ => 0,
    };
    let order = order_arg
        .map(|arg| literal_int(m, arg))
        .unwrap_or(Some(default_order))?;
    let order = usize::try_from(order).ok()?;
    let knots = entries(m, knots)?;
    if degree.checked_add(1)?.checked_mul(2)? > knots.len() {
        return None;
    }
    if member == "bspline_penalty" && order > degree {
        return None;
    }
    let n = knots.len() - degree - 1;
    let mut valid = m.alloc(Node::Lit(Scalar::Bool(true)));
    for &knot in &knots {
        let finite = build_call(m, "isfinite", &[knot]);
        valid = build_call(m, "land", &[valid, finite]);
    }
    for pair in knots.windows(2) {
        let ordered = build_call(m, "le", pair);
        valid = build_call(m, "land", &[valid, ordered]);
    }
    let nonempty = build_call(m, "lt", &[knots[degree], knots[n]]);
    valid = build_call(m, "land", &[valid, nonempty]);
    // Multiplicity may not exceed degree+1, including at the boundary.
    for i in 0..knots.len().saturating_sub(degree + 1) {
        let distinct = build_call(m, "lt", &[knots[i], knots[i + degree + 1]]);
        valid = build_call(m, "land", &[valid, distinct]);
    }
    if member == "bspline_penalty" {
        let gap = degree - order + 1;
        for i in degree + 1..n {
            if i + gap < knots.len() {
                let left = build_call(m, "gt", &[knots[i], knots[degree]]);
                let right = build_call(m, "lt", &[knots[i], knots[n]]);
                let interior = build_call(m, "land", &[left, right]);
                let distinct = build_call(m, "lt", &[knots[i], knots[i + gap]]);
                let boundary = build_call(m, "lnot", &[interior]);
                let allowed = build_call(m, "lor", &[boundary, distinct]);
                valid = build_call(m, "land", &[valid, allowed]);
            }
        }
    }
    let values = match member {
        "bspline_basis" | "bspline_derivative" => {
            let finite = build_call(m, "isfinite", &[args[0]]);
            valid = build_call(m, "land", &[valid, finite]);
            basis(m, args[0], &knots, degree, order)
        }
        "bspline_integral" => {
            for &bound in &args[..2] {
                let finite = build_call(m, "isfinite", &[bound]);
                valid = build_call(m, "land", &[valid, finite]);
            }
            integrate(m, args[0], args[1], &knots, degree)
        }
        "bspline_penalty" => penalty(m, &knots, degree, order),
        _ => return None,
    };
    let nan = lit(m, f64::NAN);
    let values: Vec<_> = values
        .into_iter()
        .map(|value| choose(m, valid, value, nan))
        .collect();
    if member == "bspline_penalty" {
        let rows: Vec<_> = values
            .chunks(n)
            .map(|row| build_call(m, "vector", row))
            .collect();
        Some((build_call(m, "vector", &rows), valid))
    } else {
        Some((build_call(m, "vector", &values), valid))
    }
}

fn basis(m: &mut Module, x: NodeId, knots: &[NodeId], degree: usize, order: usize) -> Vec<NodeId> {
    let zero = lit(m, 0.0);
    let one = lit(m, 1.0);
    let n = knots.len() - degree - 1;
    if order > degree {
        return vec![zero; n];
    }
    let inside_left = build_call(m, "ge", &[x, knots[degree]]);
    let inside_right = build_call(m, "le", &[x, knots[n]]);
    let inside = build_call(m, "land", &[inside_left, inside_right]);
    let endpoint_difference = sub(m, x, knots[n]);
    let endpoint = build_call(m, "iszero", &[endpoint_difference]);
    let mut values = Vec::with_capacity(knots.len() - 1);
    for i in 0..knots.len() - 1 {
        let lower = build_call(m, "ge", &[x, knots[i]]);
        let upper = build_call(m, "lt", &[x, knots[i + 1]]);
        let ordinary = build_call(m, "land", &[lower, upper]);
        // Left limit at the upper base endpoint, even for non-clamped knots.
        let lower_left = build_call(m, "lt", &[knots[i], x]);
        let upper_left = build_call(m, "le", &[x, knots[i + 1]]);
        let left_limit = build_call(m, "land", &[lower_left, upper_left]);
        let active = choose(m, endpoint, left_limit, ordinary);
        values.push(vec![choose(m, active, one, zero)]);
    }
    for p in 1..=degree {
        let mut next = Vec::with_capacity(values.len() - 1);
        let p_real = lit(m, p as f64);
        for i in 0..values.len() - 1 {
            let left_den = sub(m, knots[i + p], knots[i]);
            let right_den = sub(m, knots[i + p + 1], knots[i + 1]);
            let left_num = sub(m, x, knots[i]);
            let right_num = sub(m, knots[i + p + 1], x);
            let left = safe_ratio(m, left_num, left_den);
            let right = safe_ratio(m, right_num, right_den);
            let a = mul(m, left, values[i][0]);
            let b = mul(m, right, values[i + 1][0]);
            let mut derivatives = vec![add(m, a, b)];
            for r in 1..=order.min(p) {
                let a = safe_ratio(m, values[i][r - 1], left_den);
                let b = safe_ratio(m, values[i + 1][r - 1], right_den);
                let difference = sub(m, a, b);
                derivatives.push(mul(m, p_real, difference));
            }
            next.push(derivatives);
        }
        values = next;
    }
    values
        .into_iter()
        .map(|value| choose(m, inside, value[order], zero))
        .collect()
}

fn safe_ratio(m: &mut Module, numerator: NodeId, denominator: NodeId) -> NodeId {
    let zero = lit(m, 0.0);
    let one = lit(m, 1.0);
    let nonzero = build_call(m, "gt", &[denominator, zero]);
    let safe = choose(m, nonzero, denominator, one);
    let value = divide(m, numerator, safe);
    choose(m, nonzero, value, zero)
}

fn integrate(m: &mut Module, a: NodeId, b: NodeId, knots: &[NodeId], degree: usize) -> Vec<NodeId> {
    let n = knots.len() - degree - 1;
    let zero = lit(m, 0.0);
    let half = lit(m, 0.5);
    let mut result = vec![zero; n];
    for span in degree..n {
        // Clip each endpoint separately. A signed span width handles reversal
        // and preserves the two endpoint derivatives even when a == b.
        let left = build_call(m, "max", &[a, knots[span]]);
        let left = build_call(m, "min", &[left, knots[span + 1]]);
        let right = build_call(m, "max", &[b, knots[span]]);
        let right = build_call(m, "min", &[right, knots[span + 1]]);
        let width = sub(m, right, left);
        let radius = mul(m, half, width);
        let center = add(m, left, radius);
        for (node, weight) in gauss_legendre((degree + 2) / 2) {
            let node = lit(m, node);
            let offset = mul(m, radius, node);
            let x = add(m, center, offset);
            let weight = lit(m, weight);
            let scale = mul(m, radius, weight);
            for (target, value) in result.iter_mut().zip(basis(m, x, knots, degree, 0)) {
                let contribution = mul(m, scale, value);
                *target = add(m, *target, contribution);
            }
        }
    }
    result
}

fn penalty(m: &mut Module, knots: &[NodeId], degree: usize, order: usize) -> Vec<NodeId> {
    let n = knots.len() - degree - 1;
    let zero = lit(m, 0.0);
    let half = lit(m, 0.5);
    let mut result = vec![zero; n * n];
    if order > degree {
        return result;
    }
    for span in degree..n {
        let width = sub(m, knots[span + 1], knots[span]);
        let radius = mul(m, half, width);
        let center = add(m, knots[span], radius);
        for (node, weight) in gauss_legendre(degree - order + 1) {
            let node = lit(m, node);
            let offset = mul(m, radius, node);
            let x = add(m, center, offset);
            let weight = lit(m, weight);
            let scale = mul(m, radius, weight);
            let values = basis(m, x, knots, degree, order);
            for i in 0..n {
                for j in i..n {
                    let product = mul(m, values[i], values[j]);
                    let contribution = mul(m, scale, product);
                    let value = add(m, result[i * n + j], contribution);
                    result[i * n + j] = value;
                    result[j * n + i] = value;
                }
            }
        }
    }
    result
}

/// The n-point rule integrates a polynomial of degree <=2n-1 exactly.
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut derivative = 0.0;
            for _ in 0..32 {
                let mut p0 = 1.0;
                let mut p1 = z;
                for k in 2..=n {
                    let p = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p;
                }
                let (pn, previous) = if n == 1 { (z, 1.0) } else { (p1, p0) };
                derivative = n as f64 * (z * pn - previous) / (z * z - 1.0);
                let next = z - pn / derivative;
                if (next - z).abs() <= 2.0 * f64::EPSILON {
                    z = next;
                    break;
                }
                z = next;
            }
            (z, 2.0 / ((1.0 - z * z) * derivative * derivative))
        })
        .collect()
}
