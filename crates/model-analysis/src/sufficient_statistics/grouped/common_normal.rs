//! Occupied Normal means with one scale share their within-group residual energy.

use super::{Parameter, gather_parts};
use crate::sufficient_statistics::multivariate::{shared, vector};
use crate::sufficient_statistics::{
    Expr, Factor, Reducer, fixed_values, map, materialize, num, op, resolve_record_field,
};
use flatppl_core::{Dim, Module, NodeId, ScalarType, Symbol, Type};

/// Keep the rounding error of a partial sum when adding terms of different magnitude.
fn compensated_sum(terms: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut correction = 0.0;
    for term in terms {
        let next = sum + term;
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    (sum, correction)
}

// Bound the number of rounded additions even on targets that reduce each
// contiguous vector sequentially. This energy is computed once for all groups.
fn sum_squares(values: Expr, mut length: usize) -> Expr {
    const BLOCK: usize = 16;
    let mut terms = map("mul", [values.clone(), values]);
    while length > BLOCK {
        let blocks = length.div_ceil(BLOCK);
        let padding = blocks * BLOCK - length;
        if padding > 0 {
            terms = op(
                "cat",
                [terms, op("fill", [num(0.0), Expr::Integer(padding as i64)])],
            );
        }
        let shape = vector([Expr::Integer(BLOCK as i64)]);
        terms = map("sum", [op("splitblocks", [terms, shape])]);
        length = blocks;
    }
    op("sum", [terms])
}

// The two quadratics share the scale adjoint before the reciprocal is
// differentiated. Outside the normal arithmetic range keep direct quotients.
fn f32_reciprocal(m: &mut Module, sigma: &Expr) -> (Expr, Expr) {
    let range = 2.0_f64.powi(30);
    let regular = shared(
        m,
        op(
            "land",
            [
                op("ge", [sigma.clone(), num(1.0 / range)]),
                op("le", [sigma.clone(), num(range)]),
            ],
        ),
    );
    let safe = op("ifelse", [regular.clone(), sigma.clone(), num(1.0)]);
    (regular, shared(m, op("divide", [num(1.0), safe])))
}

fn standardize(
    value: Expr,
    sigma: &Expr,
    reciprocal: Option<&(Expr, Expr)>,
    apply: fn(&'static str, Vec<Expr>) -> Expr,
) -> Expr {
    let Some((regular, inverse)) = reciprocal else {
        return apply("divide", vec![value, sigma.clone()]);
    };
    // With at most 2^24 rows, |x| <= 2^32 and sigma >= 2^-30 bound
    // sum(n*x^2/sigma) by 2^118 and (x/sigma)^2 by 2^124 in binary32.
    let magnitude = apply("abs", vec![value.clone()]);
    let use_inverse = apply(
        "land",
        vec![
            regular.clone(),
            apply("le", vec![magnitude, num(2.0_f64.powi(32))]),
        ],
    );
    let choose = |a, b| apply("ifelse", vec![use_inverse.clone(), a, b]);
    let fast = apply(
        "mul",
        vec![choose(value.clone(), num(0.0)), inverse.clone()],
    );
    let direct = apply("divide", vec![choose(num(0.0), value), sigma.clone()]);
    apply("add", vec![fast, direct])
}

pub(super) fn reduce(
    r: &mut Reducer,
    m: &mut Module,
    f: &Factor,
    parameters: &[(Symbol, Parameter)],
    occupied: &[(Vec<i64>, Vec<usize>)],
) -> Option<NodeId> {
    if m.resolve(f.constructor) != "Normal" || parameters.len() != 2 {
        return None;
    }
    let mu = parameters
        .iter()
        .position(|(name, _)| m.resolve(*name) == "mu")?;
    let Parameter::Gather { array, .. } = parameters[mu].1 else {
        return None;
    };
    let (sigma_name, Parameter::Shared(sigma)) = parameters[1 - mu] else {
        return None;
    };
    if m.resolve(sigma_name) != "sigma" {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(array)? else {
        return None;
    };
    let [Dim::Static(groups)] = **shape else {
        return None;
    };
    if !matches!(**elem, Type::Scalar(ScalarType::Real))
        || !matches!(m.type_of(f.observed), Some(Type::Array { elem, .. }) if **elem == Type::Scalar(ScalarType::Real))
    {
        return None;
    }
    let (_, selector, origin) = gather_parts(m, f.parameters[mu].1, r)?;
    let selector = resolve_record_field(m, selector, &r.boundary);
    let observed = resolve_record_field(m, f.observed, &r.boundary);
    let values = fixed_values(m, observed, &r.boundary)?;
    let n = values.len();
    // Adjacent distinct f32 observations must remain at least N normal units
    // apart so that the target's mean correction can divide their gap by N.
    let smallest = 2.0 * n as f64 * f64::from(f32::MIN_POSITIVE) / f64::from(f32::EPSILON);
    if n > (1 << 24)
        || !values.iter().all(|x| {
            x.is_finite()
                && x.abs() <= f64::from(f32::MAX.sqrt() / 8.0)
                && (*x == 0.0 || x.abs() >= smallest)
        })
    {
        return None;
    }

    let mut ordered = occupied.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|(key, _)| key[0]);
    let mut labels = vec![0_i64; n];
    let rounded = values
        .iter()
        .map(|&y| f64::from(y as f32))
        .collect::<Vec<_>>();
    let mut centers = Vec::new();
    let mut low_centers = Vec::new();
    let mut minima = Vec::new();
    let mut maxima = Vec::new();
    for (j, (_, rows)) in ordered.iter().enumerate() {
        let first = values[rows[0]];
        let (mut lo, mut hi) = (first, first);
        for &i in rows {
            labels[i] = j as i64 + 1;
            lo = lo.min(values[i]);
            hi = hi.max(values[i]);
        }
        let count = rows.len() as f64;
        let (center, low) = if lo.is_sign_negative() == hi.is_sign_negative()
            && lo.abs().max(hi.abs()) <= 2.0 * lo.abs().min(hi.abs())
        {
            // Every row subtracts a nearby reference exactly in either target.
            let (offset, error) = compensated_sum(rows.iter().map(|&i| values[i] - first));
            (first + (offset + error) / count, None)
        } else {
            let (sum, error) = compensated_sum(rows.iter().map(|&i| rounded[i]));
            let center = f64::from(((sum + error) / count) as f32);
            // A binary32 center times a 24-bit integer is exact in binary64.
            (center, Some(((sum - center * count) + error) / count))
        };
        centers.push(num(center));
        low_centers.push(low);
        minima.push(lo);
        maxima.push(hi);
    }
    let origin = materialize(m, Expr::Integer(origin));
    let key = [observed, selector, origin];
    let centers = shared(m, vector(centers));
    let labels = shared(m, vector(labels.into_iter().map(Expr::Integer)));
    let by_row = |groups: Expr| op("get", [groups, labels.clone()]);
    let residual = shared(
        m,
        map("sub", [Expr::Node(observed), by_row(centers.clone())]),
    );
    // y = f32(y) + (y - f32(y)). The second term vanishes on a binary32
    // target and is an exact subtraction on a binary64 target. Accumulating
    // it keeps a small group mean after large positive and negative observations.
    let rounding_errors = shared(
        m,
        map(
            "sub",
            [
                Expr::Node(observed),
                vector(rounded.iter().copied().map(num)),
            ],
        ),
    );
    let corrections = vector(ordered.iter().enumerate().map(|(j, (_, rows))| {
        if minima[j] == maxima[j] {
            return num(0.0);
        }
        let indices = vector(rows.iter().map(|&i| Expr::Integer(i as i64 + 1)));
        let mean = |array| {
            let values = op("get", [array, indices]);
            op("sum", [map("divide", [values, num(rows.len() as f64)])])
        };
        match low_centers[j] {
            None => mean(residual.clone()),
            Some(low) if rows.iter().all(|&i| values[i] == rounded[i]) => num(low),
            Some(low) => op("add", [num(low), mean(rounding_errors.clone())]),
        }
    }));
    let corrections = r.summary_for(m, &key, "grouped_normal_corrections", corrections);
    let within = shared(m, map("sub", [residual, by_row(corrections.clone())]));
    let scale = shared(m, op("maximum", [map("abs", [within.clone()])]));
    let safe = op(
        "ifelse",
        [op("iszero", [scale.clone()]), num(1.0), scale.clone()],
    );
    let standardized = shared(m, map("divide", [within, safe]));
    // Choose a power of two at least sqrt(N). The squared ratio below can
    // overflow only when the average squared residual already overflows.
    let divisor = f64::from(1_u32 << n.next_power_of_two().ilog2().div_ceil(2));
    let scale = r.summary_for(m, &key, "grouped_normal_scale", scale);
    let weight = sum_squares(standardized, n);
    let weight = r.summary_for(
        m,
        &key,
        "grouped_normal_weight",
        op("mul", [num(divisor * divisor), weight]),
    );

    let means = if groups as usize == ordered.len() {
        Expr::Node(array)
    } else {
        op(
            "get",
            [
                Expr::Node(array),
                vector(ordered.iter().map(|(key, _)| Expr::Integer(key[0]))),
            ],
        )
    };
    let sigma_value = Expr::Node(sigma);
    let reciprocal = (r.policy == crate::sufficient_statistics::ReductionPolicy::SharedNormalScale)
        .then(|| f32_reciprocal(m, &sigma_value));
    let delta = map("add", [map("sub", [centers, means.clone()]), corrections]);
    let z = shared(
        m,
        standardize(delta, &sigma_value, reciprocal.as_ref(), map),
    );
    let count = vector(ordered.iter().map(|(_, rows)| num(rows.len() as f64)));
    let loss = map(
        "mul",
        [count, map("mul", [num(-0.5), map("mul", [z.clone(), z])])],
    );

    // A direct Normal squares each standardized observation before multiplying
    // by 1/2. Retain its overflow and the derivative of the centered score.
    let distance = map(
        "max",
        [
            map(
                "abs",
                [map(
                    "sub",
                    [vector(minima.into_iter().map(num)), means.clone()],
                )],
            ),
            map(
                "abs",
                [map("sub", [vector(maxima.into_iter().map(num)), means])],
            ),
        ],
    );
    let extreme = shared(m, map("divide", [distance, sigma_value.clone()]));
    let overflow = map("isinf", [map("mul", [extreme.clone(), extreme])]);
    let correction = map(
        "ifelse",
        [overflow, op("neg", [Expr::Constant("inf")]), num(0.0)],
    );
    let loss = op("sum", [map("add", [loss, correction])]);
    let zero = materialize(m, num(0.0));
    let anchor = r.anchor(
        m,
        f,
        &[(parameters[mu].0, zero), (sigma_name, sigma)],
        num(0.0),
    );
    let z = shared(
        m,
        op(
            "divide",
            [
                standardize(scale, &sigma_value, reciprocal.as_ref(), op),
                num(divisor),
            ],
        ),
    );
    let within = op(
        "mul",
        [weight, op("mul", [num(-0.5), op("mul", [z.clone(), z])])],
    );
    Some(materialize(
        m,
        op(
            "add",
            [
                op("add", [op("mul", [num(n as f64), anchor]), loss]),
                within,
            ],
        ),
    ))
}
