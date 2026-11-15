//! Runtime count partitions retain native kernels and endpoint derivatives.

use super::grouped::factor;
use super::multivariate::{held_cells, shared, vector};
use super::runtime_partition::Partition;
use super::{Expr, Reducer, map, materialize, num, op, resolve};
use flatppl_core::{Module, Node, NodeId, Scalar, ScalarType};

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    if !r.preparing {
        return None;
    }
    let f = factor(m, id, r)?;
    reduce_factor(r, m, &f)
}

pub(super) fn reduce_factor(r: &mut Reducer, m: &mut Module, f: &super::Factor) -> Option<NodeId> {
    if !r.preparing {
        return None;
    }
    if !matches!(m.resolve(f.constructor), "Binomial" | "Geometric") {
        return None;
    }
    let partition = Partition::new(m, f, r, ScalarType::Integer)?;
    if m.resolve(f.constructor) == "Geometric" {
        return geometric(r, m, f, &partition);
    }
    let (counts, outcomes, trials) = binomial_counts(r, m, f, &partition)?;
    let bins = partition.groups * outcomes;
    let axes = op(
        "indicesof",
        [op(
            "fill",
            [Expr::Integer(0), Expr::Integer(i64::from(bins))],
        )],
    );
    let cell = op("sub", [Expr::Local, Expr::Integer(1)]);
    let group = op(
        "add",
        [
            op("div", [cell.clone(), Expr::Integer(i64::from(outcomes))]),
            Expr::Integer(1),
        ],
    );
    let outcome = op("mod", [cell, Expr::Integer(i64::from(outcomes))]);
    let count = op("get", [Expr::Held(partition.parameters.len()), Expr::Local]);
    let empty = op("iszero", [count.clone()]);
    let mut selected = Vec::new();
    for (i, &(name, _, gathered)) in partition.parameters.iter().enumerate() {
        let value = if gathered {
            op("get", [Expr::Held(i), group.clone()])
        } else {
            Expr::Held(i)
        };
        let fallback = if m.resolve(name) == "p" {
            num(0.5)
        } else {
            Expr::Integer(trials)
        };
        selected.push((
            name,
            materialize(m, op("ifelse", [empty.clone(), fallback, value])),
        ));
    }
    let point = op(
        "ifelse",
        [
            op("equal", [outcome.clone(), Expr::Integer(trials + 1)]),
            Expr::Integer(-1),
            outcome,
        ],
    );
    let point = op("ifelse", [empty, Expr::Integer(0), point]);
    let anchor = r.anchor(m, f, &selected, point);
    let anchor = shared(m, anchor);
    let terms = op("mul", [count, anchor]);
    let mut held = partition
        .parameters
        .iter()
        .map(|(_, value, _)| Expr::Node(*value))
        .collect::<Vec<_>>();
    held.push(counts);
    r.count += 1;
    r.rule = Some(("runtime grouped Binomial outcomes", None));
    Some(materialize(m, op("sum", [held_cells(terms, axes, held)])))
}

pub(super) fn binomial_counts(
    r: &mut Reducer,
    m: &mut Module,
    f: &super::Factor,
    partition: &Partition,
) -> Option<(Expr, u32, i64)> {
    let trials = f
        .parameters
        .iter()
        .find(|(name, _)| m.resolve(*name) == "n")?;
    let trials = r.scalar(m, trials.1)?;
    if !r.data_only(m, trials) {
        return None;
    }
    let Node::Lit(Scalar::Int(trials)) = *m.node(resolve(m, trials, &r.boundary)) else {
        return None;
    };
    if !(1..=63).contains(&trials) {
        return None;
    }
    let outcomes = trials as u32 + 2;
    if partition.groups * outcomes > 256 {
        return None;
    }
    let checked = partition.checked(m);
    // One extra outcome holds every off-support integer. Clamp before adding
    // the group code, so i64 extremes never overflow integer arithmetic.
    let y = Expr::Node(f.observed);
    let supported = map(
        "land",
        [
            map("ge", [y.clone(), Expr::Integer(0)]),
            map("le", [y.clone(), Expr::Integer(trials)]),
        ],
    );
    let outcome = map("ifelse", [supported, y, Expr::Integer(trials + 1)]);
    let code = map(
        "add",
        [
            map(
                "mul",
                [
                    map("sub", [checked, Expr::Integer(1)]),
                    Expr::Integer(i64::from(outcomes)),
                ],
            ),
            outcome,
        ],
    );
    let mut keys = partition.keys(m, f.observed).to_vec();
    keys.push(materialize(m, Expr::Integer(trials)));
    let bins = partition.groups * outcomes;
    let counts = r.summary_for(
        m,
        &keys,
        "runtime_binomial_outcomes",
        op(
            "bincounts",
            [
                op("vector", (0..=bins).map(|i| num(f64::from(i) - 0.5))),
                code,
            ],
        ),
    );
    Some((counts, outcomes, trials))
}

fn geometric(
    r: &mut Reducer,
    m: &mut Module,
    f: &super::Factor,
    partition: &Partition,
) -> Option<NodeId> {
    let [(name, probability, gathered)] = *partition.parameters else {
        return None;
    };
    let (lo, hi) = super::regression::bounds(m, probability, r)?;
    // Native Geometric currently returns NaN at p=1,k=0. Keep that endpoint
    // on the original path, and prove all summed probability gradients finite.
    if lo <= 0.0
        || (lo as f32) < f32::MIN_POSITIVE
        || hi >= 1.0
        || (hi as f32) >= 1.0
        || f64::from(partition.count) * (1.0 / lo + i64::MAX as f64 / (1.0 - hi))
            > f64::from(f32::MAX) / 128.0
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
    let supported_counts = super::regression::setup(
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
    let safe = map(
        "ifelse",
        [valid.clone(), Expr::Node(f.observed), Expr::Integer(0)],
    );
    let mask = map("equal", [Expr::Held(0), Expr::Local]);
    let statistics = r.summary_for(
        m,
        &keys,
        "runtime_geometric_statistics",
        op(
            "rowstack",
            [held_cells(
                vector([
                    op("real", [op("get", [Expr::Held(3), Expr::Local])]),
                    op(
                        "sum",
                        [map(
                            "real",
                            [map("ifelse", [mask, Expr::Held(2), Expr::Integer(0)])],
                        )],
                    ),
                ]),
                partition.axes(),
                [checked, valid, safe, supported_counts],
            )],
        ),
    );
    let n = op("get", [Expr::Held(1), Expr::Local]);
    let p = if gathered {
        op("get", [Expr::Held(0), Expr::Local])
    } else {
        Expr::Held(0)
    };
    let p = shared(m, op("ifelse", [op("iszero", [n]), num(0.5), p]));
    let selected = [(name, materialize(m, p.clone()))];
    let anchor = r.anchor(m, f, &selected, Expr::Integer(0));
    let supported = op("get", [Expr::Held(2), Expr::Local, Expr::Integer(1)]);
    let total = op("get", [Expr::Held(2), Expr::Local, Expr::Integer(2)]);
    let term = op(
        "add",
        [
            op("mul", [supported, anchor]),
            op("mul", [total, op("log1p", [op("neg", [p])])]),
        ],
    );
    let result = op(
        "sum",
        [held_cells(
            term,
            partition.axes(),
            [Expr::Node(probability), counts, statistics],
        )],
    );
    let penalty = op(
        "ifelse",
        [all_valid, num(0.0), op("neg", [Expr::Constant("inf")])],
    );
    r.count += 1;
    r.rule = Some(("runtime grouped Geometric statistics", None));
    Some(materialize(m, op("add", [result, penalty])))
}
