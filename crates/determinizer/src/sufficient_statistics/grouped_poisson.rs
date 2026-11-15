//! Runtime Poisson partitions use native data-centered primitive anchors.

use super::grouped::factor;
use super::multivariate::{held_cells, shared, vector};
use super::regression::{bounds, setup};
use super::runtime_partition::Partition;
use super::{Expr, Reducer, map, materialize, num, op, relative_log};
use flatppl_core::{Module, NodeId, ScalarType};

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    if !r.preparing {
        return None;
    }
    let f = factor(m, id, r)?;
    if m.resolve(f.constructor) != "Poisson" {
        return None;
    }
    let partition = Partition::new(m, &f, r, ScalarType::Integer)?;
    let [(name, rate, gathered)] = *partition.parameters else {
        return None;
    };
    let (lo, hi) = bounds(m, rate, r)?;
    // A positive lower bound excludes the primitive's distinct zero-rate
    // adjoint, and prevents overflowing opposite-signed centered derivatives.
    let maximum_total = f64::from(partition.count) * i64::MAX as f64;
    if lo <= 0.0
        || (lo as f32) < f32::MIN_POSITIVE
        || !hi.is_finite()
        || (maximum_total / lo).max(f64::from(partition.count) * hi) > f64::from(f32::MAX) / 128.0
    {
        return None;
    }
    let keys = partition.keys(m, f.observed);
    let checked = partition.checked(m);
    let counts = partition.counts(r, m, checked.clone());
    let valid = map("ge", [Expr::Node(f.observed), Expr::Integer(0)]);
    let all_valid = r.summary(
        m,
        f.observed,
        "count_nonnegative",
        op("lall", [valid.clone()]),
    );
    let supported_counts = setup(
        m,
        op(
            "bincounts",
            [
                op(
                    "vector",
                    (0..=partition.groups).map(|i| num(f64::from(i) + 0.5)),
                ),
                map("ifelse", [valid.clone(), checked.clone(), Expr::Integer(0)]),
            ],
        ),
    );
    let safe = setup(
        m,
        map(
            "ifelse",
            [valid.clone(), Expr::Node(f.observed), Expr::Integer(0)],
        ),
    );
    let mask = map("equal", [Expr::Held(1), Expr::Local]);
    let supported = map("land", [mask.clone(), Expr::Held(3)]);
    let count = op("get", [Expr::Held(2), Expr::Local]);
    let reference = op(
        "max",
        [
            op(
                "sum",
                [map(
                    "divide",
                    [
                        map(
                            "real",
                            [map(
                                "ifelse",
                                [supported.clone(), Expr::Held(0), Expr::Integer(0)],
                            )],
                        ),
                        op("max", [count.clone(), Expr::Integer(1)]),
                    ],
                )],
            ),
            num(1.0),
        ],
    );
    let references = setup(
        m,
        held_cells(
            op("real", [reference]),
            partition.axes(),
            [
                safe.clone(),
                checked.clone(),
                supported_counts.clone(),
                valid.clone(),
            ],
        ),
    );
    let reference = op("get", [Expr::Held(4), Expr::Local]);
    let distance = map(
        "abs",
        [map(
            "sub",
            [map("real", [Expr::Held(0)]), reference.clone()],
        )],
    );
    let distance = map(
        "ifelse",
        [supported.clone(), distance, Expr::Constant("inf")],
    );
    let nearest = map(
        "land",
        [
            supported.clone(),
            map("le", [distance.clone(), op("minimum", [distance])]),
        ],
    );
    let first = op(
        "minimum",
        [map(
            "ifelse",
            [
                nearest,
                map("real", [op("indicesof", [Expr::Held(0)])]),
                num(f64::from(partition.count)),
            ],
        )],
    );
    let center = op(
        "get",
        [Expr::Held(0), op("integer", [op("round", [first])])],
    );
    let centers = r.summary_for(
        m,
        &keys,
        "runtime_poisson_centers",
        held_cells(
            center,
            partition.axes(),
            [
                safe.clone(),
                checked.clone(),
                supported_counts.clone(),
                valid.clone(),
                references.clone(),
            ],
        ),
    );
    let center = op("get", [Expr::Held(5), Expr::Local]);
    let values = map("ifelse", [supported.clone(), Expr::Held(0), center.clone()]);
    let reference_parameter = materialize(m, Expr::Held(0));
    let inner_anchor = r.anchor(m, &f, &[(name, reference_parameter)], Expr::Local);
    let scores = held_cells(inner_anchor, values.clone(), [reference.clone()]);
    let outer_reference = materialize(m, reference.clone());
    let anchor = r.anchor(m, &f, &[(name, outer_reference)], center.clone());
    let correction = op("sum", [map("sub", [scores, anchor])]);
    let residual = op(
        "sum",
        [map("sub", [map("real", [values]), op("real", [center])])],
    );
    let statistics = r.summary_for(
        m,
        &keys,
        "runtime_poisson_statistics",
        op(
            "rowstack",
            [held_cells(
                vector([
                    op("real", [reference]),
                    op("real", [count]),
                    correction,
                    op("real", [residual]),
                ]),
                partition.axes(),
                [
                    safe,
                    checked,
                    supported_counts.clone(),
                    valid,
                    references,
                    centers.clone(),
                ],
            )],
        ),
    );
    let n = op("get", [Expr::Held(1), Expr::Local]);
    let parameter = if gathered {
        op("get", [Expr::Held(0), Expr::Local])
    } else {
        Expr::Held(0)
    };
    let parameter = shared(m, op("ifelse", [op("iszero", [n]), num(1.0), parameter]));
    let parameter_id = materialize(m, parameter.clone());
    let center = op("get", [Expr::Held(2), Expr::Local]);
    let anchor = r.anchor(m, &f, &[(name, parameter_id)], center);
    let anchor = shared(m, anchor);
    let finite = op("isfinite", [anchor.clone()]);
    let at = |column| op("get", [Expr::Held(3), Expr::Local, Expr::Integer(column)]);
    let base = op("mul", [at(2), anchor]);
    let reference = at(1);
    let active = op(
        "land",
        [finite.clone(), op("lnot", [op("iszero", [at(4)])])],
    );
    let safe_rate = op("ifelse", [active, parameter, reference.clone()]);
    let corrected = op(
        "add",
        [
            base.clone(),
            op(
                "add",
                [
                    at(3),
                    op("mul", [at(4), relative_log(safe_rate, reference)]),
                ],
            ),
        ],
    );
    let term = op("ifelse", [finite, corrected, base]);
    let result = op(
        "sum",
        [held_cells(
            term,
            partition.axes(),
            [Expr::Node(rate), counts, centers, statistics],
        )],
    );
    let penalty = op(
        "ifelse",
        [all_valid, num(0.0), op("neg", [Expr::Constant("inf")])],
    );
    r.count += 1;
    r.rule = Some(("runtime grouped Poisson anchors", None));
    Some(materialize(m, op("add", [result, penalty])))
}
