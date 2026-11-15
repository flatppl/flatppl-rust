//! Explicit sorted-data preparation for scalar Laplace locations and scales.

use super::regression::setup as shared;
use super::{Expr, Reducer, map, num, op, per_cell_with_invariants};
use flatppl_core::{Dim, Module, NodeId, ScalarType, Type};

fn integer(n: u32) -> Expr {
    Expr::Integer(i64::from(n))
}

fn get(array: Expr, index: Expr) -> Expr {
    op("get", [array, index])
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<Expr> {
    if !r.preparing {
        return None;
    }
    let f = per_cell_with_invariants(m, id, &r.boundary)?;
    if m.resolve(f.constructor) != "Laplace" || !r.data_only(m, f.observed) {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(f.observed)? else {
        return None;
    };
    let [Dim::Static(n)] = **shape else {
        return None;
    };
    if !(2..=16384).contains(&n) || **elem != Type::Scalar(ScalarType::Real) {
        return None;
    }
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, value)| Some((name, r.scalar(m, value)?)))
        .collect::<Option<Vec<_>>>()?;
    let location = parameters
        .iter()
        .find(|(name, _)| m.resolve(*name) == "location")?
        .1;
    let scale = parameters
        .iter()
        .find(|(name, _)| m.resolve(*name) == "scale")?
        .1;
    let (a, b) = super::regression::bounds(m, f.observed, r)?;
    let (u, v) = super::regression::bounds(m, location, r)?;
    let (lo, hi) = super::regression::bounds(m, scale, r)?;
    let magnitude = a.abs().max(b.abs());
    let distance = 2.0 * magnitude + u.abs().max(v.abs());
    // Bound gaps, distance sums and the reverse scale derivative in f32.
    // The same proof therefore covers both supported emission precisions.
    if ![a, b, u, v, lo, hi].iter().all(|x| (*x as f32).is_finite())
        || lo <= 0.0
        || (lo as f32) * (lo as f32) < f32::MIN_POSITIVE * 128.0
        || f64::from(n) * distance > f64::from(f32::MAX) / 128.0
        || f64::from(n) * (1.0 + distance / lo) / lo > f64::from(f32::MAX) / 128.0
    {
        return None;
    }
    let y = Expr::Node(f.observed);
    let valid = r.summary(
        m,
        f.observed,
        "all_finite",
        op("lall", [map("isfinite", [y.clone()])]),
    );
    let raw_min = r.summary(m, f.observed, "minimum", op("minimum", [y.clone()]));
    let p = n.next_power_of_two();
    let positions = shared(
        m,
        map(
            "sub",
            [
                op("indicesof", [op("fill", [integer(0), integer(p)])]),
                integer(1),
            ],
        ),
    );
    let clean = map("ifelse", [map("isfinite", [y.clone()]), y, num(0.0)]);
    let mut sorted = shared(
        m,
        if p == n {
            clean
        } else {
            op(
                "cat",
                [clean, op("fill", [Expr::Constant("inf"), integer(p - n)])],
            )
        },
    );
    let mut k = 2;
    // A portable bitonic network uses existing gathers and pointwise min/max.
    // Padding is removed before gaps, so no infinity enters a distance sum.
    while k <= p {
        let mut j = k / 2;
        while j > 0 {
            let low = shared(
                m,
                map(
                    "equal",
                    [
                        map(
                            "mod",
                            [map("div", [positions.clone(), integer(j)]), integer(2)],
                        ),
                        integer(0),
                    ],
                ),
            );
            let indices = map(
                "add",
                [
                    map(
                        "ifelse",
                        [
                            low.clone(),
                            map("add", [positions.clone(), integer(j)]),
                            map("sub", [positions.clone(), integer(j)]),
                        ],
                    ),
                    integer(1),
                ],
            );
            let other = get(sorted.clone(), indices);
            let ascending = map(
                "equal",
                [
                    map(
                        "mod",
                        [map("div", [positions.clone(), integer(k)]), integer(2)],
                    ),
                    integer(0),
                ],
            );
            let choose_min = map("equal", [ascending, low]);
            sorted = shared(
                m,
                map(
                    "ifelse",
                    [
                        choose_min,
                        map("min", [sorted.clone(), other.clone()]),
                        map("max", [sorted, other]),
                    ],
                ),
            );
            j /= 2;
        }
        k *= 2;
    }
    let sorted = shared(
        m,
        get(
            sorted,
            op("indicesof", [op("fill", [integer(0), integer(n)])]),
        ),
    );
    let j = shared(
        m,
        op("indicesof", [op("fill", [integer(0), integer(n - 1)])]),
    );
    let gaps = shared(
        m,
        map(
            "sub",
            [
                get(sorted.clone(), map("add", [j.clone(), integer(1)])),
                get(sorted.clone(), j.clone()),
            ],
        ),
    );
    let fraction = map("divide", [map("real", [j.clone()]), num(f64::from(n))]);
    let left = op(
        "cat",
        [
            op("vector", [num(0.0)]),
            op("cumsum", [map("mul", [fraction, gaps.clone()])]),
        ],
    );
    let fraction = map(
        "divide",
        [
            map("sub", [num(f64::from(n)), map("real", [j])]),
            num(f64::from(n)),
        ],
    );
    let right = op(
        "cat",
        [
            op(
                "reverse",
                [op(
                    "cumsum",
                    [op("reverse", [map("mul", [fraction, gaps])])],
                )],
            ),
            op("vector", [num(0.0)]),
        ],
    );
    // A contiguous vector avoids a strided matrix-column copy at every probe.
    let table = r.summary(
        m,
        f.observed,
        "sorted_laplace_gaps",
        op("cat", [sorted, left, right]),
    );
    let mu = Expr::Node(location);
    let mut low = integer(0);
    let mut high = integer(n);
    for _ in 0..(u32::BITS - n.leading_zeros()) {
        let mid = shared(
            m,
            op("div", [op("add", [low.clone(), high.clone()]), integer(2)]),
        );
        let probe = op(
            "add",
            [op("min", [mid.clone(), integer(n - 1)]), integer(1)],
        );
        let probe = op("integer", [probe]);
        // Strict lower_bound preserves the primitive abs derivative at knots.
        let right = shared(m, op("lt", [get(table.clone(), probe), mu.clone()]));
        let active = shared(m, op("lt", [low.clone(), high.clone()]));
        let next = shared(
            m,
            op(
                "ifelse",
                [
                    op("land", [active.clone(), right.clone()]),
                    op("add", [mid.clone(), integer(1)]),
                    low,
                ],
            ),
        );
        high = shared(
            m,
            op(
                "ifelse",
                [op("land", [active, op("lnot", [right])]), mid, high],
            ),
        );
        low = next;
    }
    let k = low;
    let left_index = shared(m, op("integer", [op("max", [integer(1), k.clone()])]));
    let right_index = shared(
        m,
        op(
            "integer",
            [op("min", [integer(n), op("add", [k.clone(), integer(1)])])],
        ),
    );
    let left = op(
        "ifelse",
        [
            op("gt", [k.clone(), integer(0)]),
            op(
                "add",
                [
                    get(table.clone(), op("add", [integer(n), left_index.clone()])),
                    op(
                        "mul",
                        [
                            op("divide", [op("real", [k.clone()]), num(f64::from(n))]),
                            op("sub", [mu.clone(), get(table.clone(), left_index)]),
                        ],
                    ),
                ],
            ),
            num(0.0),
        ],
    );
    let right = op(
        "ifelse",
        [
            op("lt", [k.clone(), integer(n)]),
            op(
                "add",
                [
                    get(
                        table.clone(),
                        op("add", [integer(2 * n), right_index.clone()]),
                    ),
                    op(
                        "mul",
                        [
                            op(
                                "divide",
                                [
                                    op("sub", [num(f64::from(n)), op("real", [k])]),
                                    num(f64::from(n)),
                                ],
                            ),
                            op("sub", [get(table, right_index), mu.clone()]),
                        ],
                    ),
                ],
            ),
            num(0.0),
        ],
    );
    let anchor = r.anchor(m, &f, &parameters, mu);
    let result = op(
        "sub",
        [
            op("mul", [num(f64::from(n)), anchor]),
            op(
                "mul",
                [
                    num(f64::from(n)),
                    op("divide", [op("add", [left, right]), Expr::Node(scale)]),
                ],
            ),
        ],
    );
    let result = op(
        "ifelse",
        [
            op("isnan", [raw_min.clone()]),
            raw_min,
            op(
                "ifelse",
                [valid, result, op("neg", [Expr::Constant("inf")])],
            ),
        ],
    );
    Some(op(
        "ifelse",
        [
            op(
                "lor",
                [
                    op("isnan", [Expr::Node(location)]),
                    op("isnan", [Expr::Node(scale)]),
                ],
            ),
            op("divide", [num(0.0), num(0.0)]),
            result,
        ],
    ))
}
