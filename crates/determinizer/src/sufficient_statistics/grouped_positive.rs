//! Masked sufficient statistics for runtime positive-family partitions.

use super::grouped::factor;
use super::multivariate::{held_cells, shared, vector};
use super::regression::{bounds, setup};
use super::{Expr, Reducer, complement_relative_log, map, materialize, num, op, relative_log};
use flatppl_core::{Module, NodeId, ScalarType};

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    if !r.preparing {
        return None;
    }
    let f = factor(m, id, r)?;
    let family = m.resolve(f.constructor).to_owned();
    if !matches!(family.as_str(), "Exponential" | "Gamma" | "Beta") {
        return None;
    }
    let partition = super::runtime_partition::Partition::new(m, &f, r, ScalarType::Real)?;
    let n = partition.count;
    let parameters = &partition.parameters;
    let (lo, hi) = bounds(m, f.observed, r)?;
    // Bound both primal arithmetic and first derivatives in f32. The same
    // portable expression is also emitted in f64. Native anchors retain the
    // family's normalization instead of subtracting large log-gamma terms.
    let magnitude = lo.abs().max(hi.abs()).max(1.0);
    let mut work = magnitude;
    let mut minimum_parameter = f64::INFINITY;
    let mut maximum_parameter: f64 = 1.0;
    for &(name, value, _) in parameters {
        let (a, b) = bounds(m, value, r)?;
        if a <= 0.0 || (a as f32) < f32::MIN_POSITIVE {
            return None;
        }
        // At high concentration the native f32 log-gamma cancellation is
        // observation-dependent. Relocating its anchor needs a separate proof.
        if matches!(m.resolve(name), "shape" | "alpha" | "beta") && b > 64.0 {
            return None;
        }
        work = work.max(b * (1.0 + b.ln().abs() + magnitude)).max(1.0 / a);
        minimum_parameter = minimum_parameter.min(a);
        maximum_parameter = maximum_parameter.max(b);
    }
    if family != "Exponential" && (lo <= 0.0 || (lo as f32) < f32::MIN_POSITIVE) {
        return None;
    }
    if family == "Beta" && (hi >= 1.0 || (hi as f32) >= 1.0) {
        return None;
    }
    if family == "Exponential" && lo < 0.0 {
        return None;
    }
    work = work.max(maximum_parameter / minimum_parameter);
    if family != "Exponential" {
        work = work.max(maximum_parameter * (lo.ln().abs() + hi.ln().abs() + 1.0));
    }
    if family == "Beta" {
        work = work.max(maximum_parameter * (1.0 - hi).ln().abs());
    }
    if !work.is_finite() || f64::from(n) * work > f64::from(f32::MAX) / 128.0 {
        return None;
    }
    let keys = partition.keys(m, f.observed);
    let axes = partition.axes();
    let checked = partition.checked(m);
    let counts = partition.counts(r, m, checked.clone());
    let mask = map("equal", [Expr::Held(1), Expr::Local]);
    let count = op("get", [Expr::Held(2), Expr::Local]);
    let denominator = op("max", [count.clone(), Expr::Integer(1)]);
    let fallback = num(if family == "Beta" { 0.5 } else { 1.0 });
    let center = if family == "Exponential" {
        num(0.0)
    } else {
        op(
            "ifelse",
            [
                op("iszero", [count.clone()]),
                fallback.clone(),
                op(
                    "sum",
                    [map(
                        "divide",
                        [
                            map("ifelse", [mask.clone(), Expr::Held(0), num(0.0)]),
                            denominator,
                        ],
                    )],
                ),
            ],
        )
    };
    let center = if family == "Exponential" {
        center
    } else {
        op("max", [num(lo), op("min", [num(hi), center])])
    };
    let centers = setup(
        m,
        held_cells(
            center,
            axes.clone(),
            [Expr::Node(f.observed), checked.clone(), counts.clone()],
        ),
    );
    let center = op("get", [Expr::Held(3), Expr::Local]);
    // Substitute before logs and subtraction. An absent row contributes exact
    // zero and cannot poison another group's summary or derivative.
    let values = map("ifelse", [mask.clone(), Expr::Held(0), center.clone()]);
    let first = if family == "Exponential" {
        op(
            "sum",
            [map("ifelse", [mask.clone(), Expr::Held(0), num(0.0)])],
        )
    } else {
        op("sum", [relative_log(values.clone(), center.clone())])
    };
    let second = if family == "Gamma" {
        op("sum", [map("sub", [values, center.clone()])])
    } else if family == "Beta" {
        op("sum", [complement_relative_log(values, center.clone())])
    } else {
        num(0.0)
    };
    let stats = r.summary_for(
        m,
        &keys,
        match family.as_str() {
            "Gamma" => "runtime_gamma_statistics",
            "Beta" => "runtime_beta_statistics",
            _ => "runtime_exponential_statistics",
        },
        op(
            "rowstack",
            [held_cells(
                vector([center, first, second]),
                axes.clone(),
                [Expr::Node(f.observed), checked, counts.clone(), centers],
            )],
        ),
    );
    let count_slot = parameters.len();
    let stats_slot = count_slot + 1;
    let count = op("get", [Expr::Held(count_slot), Expr::Local]);
    let empty = op("iszero", [count.clone()]);
    let mut selected = Vec::new();
    for (i, &(name, _, gathered)) in parameters.iter().enumerate() {
        let value = if gathered {
            op("get", [Expr::Held(i), Expr::Local])
        } else {
            Expr::Held(i)
        };
        let value = shared(m, op("ifelse", [empty.clone(), num(1.0), value]));
        selected.push((name, materialize(m, value)));
    }
    let at = |column| {
        op(
            "get",
            [Expr::Held(stats_slot), Expr::Local, Expr::Integer(column)],
        )
    };
    let anchor = r.anchor(m, &f, &selected, at(1));
    let parameter = |name: &str| {
        selected
            .iter()
            .find(|(key, _)| m.resolve(*key) == name)
            .map(|(_, id)| Expr::Node(*id))
    };
    let result = op("mul", [count, anchor]);
    let result = match family.as_str() {
        "Gamma" => op(
            "sub",
            [
                op(
                    "add",
                    [
                        result,
                        op("mul", [op("sub", [parameter("shape")?, num(1.0)]), at(2)]),
                    ],
                ),
                op("mul", [parameter("rate")?, at(3)]),
            ],
        ),
        "Beta" => op(
            "add",
            [
                result,
                op(
                    "add",
                    [
                        op("mul", [op("sub", [parameter("alpha")?, num(1.0)]), at(2)]),
                        op("mul", [op("sub", [parameter("beta")?, num(1.0)]), at(3)]),
                    ],
                ),
            ],
        ),
        _ => op("sub", [result, op("mul", [parameter("rate")?, at(2)])]),
    };
    let mut held = parameters
        .iter()
        .map(|(_, id, _)| Expr::Node(*id))
        .collect::<Vec<_>>();
    held.extend([counts, stats]);
    let result = op("sum", [held_cells(result, axes, held)]);
    r.count += 1;
    r.rule = Some(("runtime grouped positive-family statistics", None));
    Some(materialize(m, result))
}
