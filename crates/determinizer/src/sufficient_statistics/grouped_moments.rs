//! Runtime partitions with bounded Gaussian moments and portable masked setup.

use super::grouped::{factor, fixed_indices, gather_parts};
use super::multivariate::{entry, held_cells, shared, vector};
use super::{Expr, Factor, Reducer, map, materialize, num, op, resolve};
use flatppl_core::{Dim, Module, NodeId, ScalarType, Type};

fn vector_size(m: &Module, id: NodeId, kind: ScalarType) -> Option<u32> {
    match m.type_of(id)? {
        Type::Array { shape, elem } if **elem == Type::Scalar(kind) => match **shape {
            [Dim::Static(size)] => Some(size),
            _ => None,
        },
        _ => None,
    }
}

fn parameter_range(m: &Module, id: NodeId, r: &Reducer) -> Option<(f64, f64)> {
    let (lo, hi) = super::regression::bounds(m, id, r)?;
    [lo, hi]
        .iter()
        .all(|x| (*x as f32).is_finite())
        .then_some((lo, hi))
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    if !r.preparing {
        return None;
    }
    let f = factor(m, id, r)?;
    let logarithmic = match m.resolve(f.constructor) {
        "Normal" => false,
        "LogNormal" => true,
        _ => return None,
    };
    let count = vector_size(m, f.observed, ScalarType::Real)?;
    if count < 2 || !r.data_only(m, f.observed) {
        return None;
    }
    let mut partition = None;
    let mut parameters = Vec::new();
    for &(name, value) in &f.parameters {
        if let Some(scalar) = r.scalar(m, value) {
            parameters.push((name, scalar, false));
            continue;
        }
        let (array, indices, origin) = gather_parts(m, value, r)?;
        let groups = vector_size(m, array, ScalarType::Real)?;
        let key = (resolve(m, indices, &r.boundary), origin, groups);
        if partition.is_some_and(|previous: (NodeId, i64, u32)| {
            previous.1 != origin
                || previous.2 != groups
                || r.data_key(m, previous.0) != r.data_key(m, key.0)
        }) || vector_size(m, indices, ScalarType::Integer)? != count
            || !r.data_only(m, indices)
            || fixed_indices(m, indices, r).is_some()
        {
            return None;
        }
        partition = Some(key);
        parameters.push((name, array, true));
    }
    let (indices, origin, groups) = partition?;
    // This setup uses G masked reductions over N observations. Bound its
    // workspace, and keep counts and first-position probes exact in f32.
    if !(1..=64).contains(&groups)
        || groups >= count
        || u64::from(groups) * u64::from(count) > 1 << 22
    {
        return None;
    }
    let mu = parameters
        .iter()
        .position(|(name, ..)| m.resolve(*name) == "mu")?;
    let sigma = parameters
        .iter()
        .position(|(name, ..)| m.resolve(*name) == "sigma")?;
    let (a, b) = parameter_range(m, f.observed, r)?;
    let (a, b) = if logarithmic {
        if a <= 0.0 || (a as f32) < f32::MIN_POSITIVE {
            return None;
        }
        (a.ln(), b.ln())
    } else {
        (a, b)
    };
    let (u, v) = parameter_range(m, parameters[mu].1, r)?;
    let (lo, hi) = parameter_range(m, parameters[sigma].1, r)?;
    let magnitude = a.abs().max(b.abs());
    let difference = 2.0 * magnitude + u.abs().max(v.abs());
    let residual = difference / lo;
    let minimum_square = (lo as f32) * (lo as f32);
    let maximum_square = (hi as f32) * (hi as f32);
    // A margin covers sums, centered cross terms and both precisions. Merely
    // finite intervals do not prevent inf-inf in an anchored quadratic.
    if lo <= 0.0
        || (lo as f32) < f32::MIN_POSITIVE
        || !(hi as f32).is_finite()
        || !(2.0 * magnitude as f32).is_finite()
        || minimum_square < f32::MIN_POSITIVE * 128.0
        || maximum_square > f32::MAX / 128.0
        || !residual.is_finite()
        || f64::from(count) * residual * residual > f64::from(f32::MAX) / 128.0
        || f64::from(count) * (1.0 + residual * residual) / lo > f64::from(f32::MAX) / 128.0
        || f64::from(count) * difference * (1.0 + residual) > f64::from(f32::MAX) / 128.0
    {
        return None;
    }
    let size = materialize(m, Expr::Integer(i64::from(groups)));
    let base = materialize(m, Expr::Integer(origin));
    let axes = op(
        "indicesof",
        [op("fill", [Expr::Integer(0), Expr::Node(size)])],
    );
    let labels = op(
        if origin == 0 {
            "indicesof0"
        } else {
            "indicesof"
        },
        [op("fill", [Expr::Integer(0), Expr::Node(size)])],
    );
    let checked = op(
        if origin == 0 { "get0" } else { "get" },
        [labels, Expr::Node(indices)],
    );
    let checked = shared(m, map("add", [checked, Expr::Integer(1 - origin)]));
    let counts = r.summary_for(
        m,
        &[indices, size, base],
        "runtime_group_counts",
        op(
            "bincounts",
            [
                op("vector", (0..=groups).map(|i| num(f64::from(i) + 0.5))),
                checked.clone(),
            ],
        ),
    );
    let x = Expr::Node(f.observed);
    let valid = if logarithmic {
        map(
            "land",
            [
                map("isfinite", [x.clone()]),
                map("gt", [x.clone(), num(0.0)]),
            ],
        )
    } else {
        map("isfinite", [x.clone()])
    };
    let all_valid = r.summary(
        m,
        f.observed,
        if logarithmic {
            "all_positive_finite"
        } else {
            "all_finite"
        },
        op("lall", [valid.clone()]),
    );
    let safe = shared(
        m,
        map(
            "ifelse",
            [valid, x.clone(), num(if logarithmic { 1.0 } else { 0.0 })],
        ),
    );
    let y = if logarithmic {
        shared(m, map("log", [safe]))
    } else {
        safe
    };
    let mask = shared(m, map("equal", [Expr::Held(1), Expr::Local]));
    let n = op("get", [Expr::Held(2), Expr::Local]);
    let empty = op("iszero", [n.clone()]);
    let safe_n = op("max", [n.clone(), Expr::Integer(1)]);
    // N<=2^22 makes these real position probes exact. N is a valid fallback
    // position for an empty group, so no sentinel creates a new bounds error.
    let positions = map("real", [op("indicesof", [Expr::Held(0)])]);
    let first_index = op(
        "integer",
        [op(
            "round",
            [op(
                "minimum",
                [map(
                    "ifelse",
                    [mask.clone(), positions, num(f64::from(count))],
                )],
            )],
        )],
    );
    let first = op("get", [Expr::Held(0), first_index]);
    let values = map("ifelse", [mask.clone(), Expr::Held(0), num(0.0)]);
    let mixed = op(
        "land",
        [
            op("lt", [op("minimum", [values.clone()]), num(0.0)]),
            op("gt", [op("maximum", [values]), num(0.0)]),
        ],
    );
    let first = shared(
        m,
        op(
            "ifelse",
            [op("lor", [empty.clone(), mixed]), num(0.0), first],
        ),
    );
    let shifted = map(
        "sub",
        [
            map("ifelse", [mask.clone(), Expr::Held(0), first.clone()]),
            first.clone(),
        ],
    );
    let center = op(
        "add",
        [first, op("sum", [map("divide", [shifted, safe_n.clone()])])],
    );
    let centers = shared(
        m,
        held_cells(
            center,
            axes.clone(),
            [y.clone(), checked.clone(), counts.clone()],
        ),
    );
    let center = op("get", [Expr::Held(3), Expr::Local]);
    let residuals = shared(
        m,
        map(
            "sub",
            [
                map("ifelse", [mask, Expr::Held(0), center.clone()]),
                center.clone(),
            ],
        ),
    );
    let residual_mean = op("sum", [map("divide", [residuals.clone(), safe_n])]);
    let scale = shared(m, op("maximum", [map("abs", [residuals.clone()])]));
    let scaled = shared(
        m,
        map(
            "divide",
            [
                residuals,
                op(
                    "ifelse",
                    [op("iszero", [scale.clone()]), num(1.0), scale.clone()],
                ),
            ],
        ),
    );
    let spread = op("sqrt", [op("sum", [map("mul", [scaled.clone(), scaled])])]);
    let extra = if logarithmic {
        op("sum", [y.clone()])
    } else {
        op("minimum", [x])
    };
    let moments = r.summary_for(
        m,
        &[f.observed, indices, size, base],
        if logarithmic {
            "runtime_log_normal_moments"
        } else {
            "runtime_normal_moments"
        },
        op(
            "rowstack",
            [held_cells(
                vector([center, residual_mean, scale, spread, Expr::Held(4)]),
                axes.clone(),
                [y, checked, counts.clone(), centers, extra],
            )],
        ),
    );
    let n = op("get", [Expr::Held(2), Expr::Local]);
    let empty = op("iszero", [n.clone()]);
    let parameter = |i: usize| {
        let value = if parameters[i].2 {
            op("get", [Expr::Held(i), Expr::Local])
        } else {
            Expr::Held(i)
        };
        op(
            "ifelse",
            [
                empty.clone(),
                num(if i == sigma { 1.0 } else { 0.0 }),
                value,
            ],
        )
    };
    let center = op("get", [Expr::Held(3), Expr::Local, Expr::Integer(1)]);
    let residual_mean = op("get", [Expr::Held(3), Expr::Local, Expr::Integer(2)]);
    let scale = op("get", [Expr::Held(3), Expr::Local, Expr::Integer(3)]);
    let spread = op("get", [Expr::Held(3), Expr::Local, Expr::Integer(4)]);
    let safe_mu = shared(m, parameter(mu));
    let safe_sigma = shared(m, parameter(sigma));
    let normal = Factor {
        constructor: m.intern("Normal"),
        parameters: vec![],
        observed: f.observed,
    };
    let selected = parameters
        .iter()
        .enumerate()
        .map(|(i, (name, ..))| {
            (
                *name,
                materialize(
                    m,
                    if i == mu {
                        safe_mu.clone()
                    } else {
                        safe_sigma.clone()
                    },
                ),
            )
        })
        .collect::<Vec<_>>();
    let anchor = r.anchor(m, &normal, &selected, center.clone());
    let z = op("divide", [op("sub", [center, safe_mu]), safe_sigma.clone()]);
    let cross = op(
        "mul",
        [
            op("mul", [num(2.0), n.clone()]),
            op(
                "mul",
                [z, op("divide", [residual_mean.clone(), safe_sigma.clone()])],
            ),
        ],
    );
    let cross = op("ifelse", [op("iszero", [residual_mean]), num(0.0), cross]);
    let spread = op("mul", [op("divide", [scale, safe_sigma]), spread]);
    let correction = op(
        "mul",
        [
            num(0.5),
            op("add", [cross, op("mul", [spread.clone(), spread])]),
        ],
    );
    let terms = held_cells(
        op("sub", [op("mul", [n, anchor]), correction]),
        axes,
        [
            Expr::Node(parameters[0].1),
            Expr::Node(parameters[1].1),
            counts,
            moments.clone(),
        ],
    );
    let result = op("sum", [terms]);
    let extra = entry(moments, 0, 4);
    let result = if logarithmic {
        op("sub", [result, extra.clone()])
    } else {
        result
    };
    let result = op(
        "ifelse",
        [all_valid, result, op("neg", [Expr::Constant("inf")])],
    );
    let result = if logarithmic {
        result
    } else {
        op("ifelse", [op("isnan", [extra.clone()]), extra, result])
    };
    r.count += 1;
    r.rule = Some(("runtime grouped Gaussian moments", None));
    Some(materialize(m, result))
}
