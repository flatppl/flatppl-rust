//! Vector and matrix event summaries. The outer axis is the observation axis.

use super::{Expr, Reducer, call, data_only, materialize, num, op, per_cell, resolve};
use flatppl_core::{Dim, Module, Node, NodeId, RefNs, Scalar, ScalarType, Type, ValueSet};

pub(super) fn cells(body: Expr, data: Expr) -> Expr {
    held_cells(body, data, [])
}

/// Pass invariant values in singleton cells, outside the mapped function's scope.
pub(super) fn held_cells(body: Expr, data: Expr, held: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::Cells(Box::new(body), Box::new(data), held.into_iter().collect())
}

pub(super) fn get(value: Expr, index: usize) -> Expr {
    op("get", [value, Expr::Integer(index as i64 + 1)])
}

pub(super) fn entry(value: Expr, row: usize, col: usize) -> Expr {
    op(
        "get",
        [
            value,
            Expr::Integer(row as i64 + 1),
            Expr::Integer(col as i64 + 1),
        ],
    )
}

pub(super) fn shared(m: &mut Module, expression: Expr) -> Expr {
    Expr::Node(materialize(m, expression))
}

pub(super) fn vector(values: impl IntoIterator<Item = Expr>) -> Expr {
    op("vector", values)
}

pub(super) fn dot(a: Expr, b: Expr) -> Expr {
    op("sum", [super::map("mul", [a, b])])
}

/// QR of bounded-size blocks avoids both Gram cancellation and an N-by-N Q.
/// A scan needs no associativity assumption about the signs of QR factors.
pub(super) fn residual_factor(rows: Expr, count: usize, dimension: usize) -> Expr {
    let blocks = count.div_ceil(dimension);
    let padding = blocks * dimension - count;
    let zero_row = op("fill", [num(0.0), Expr::Integer(dimension as i64)]);
    let rows = if padding == 0 {
        rows
    } else {
        op(
            "cat",
            [rows, op("fill", [zero_row, Expr::Integer(padding as i64)])],
        )
    };
    // A small sample already fits the factor budget. Keep its rows exactly.
    if count <= dimension {
        return op("rowstack", [rows]);
    }
    let block_shape = vector([
        Expr::Integer(dimension as i64),
        Expr::Integer(dimension as i64),
    ]);
    let blocks_data = op(
        "get",
        [
            op("splitblocks", [op("rowstack", [rows]), block_shape.clone()]),
            Expr::Constant("all"),
            Expr::Integer(1),
        ],
    );
    let stacked = vector([Expr::Local, Expr::Next].into_iter().flat_map(|matrix| {
        (0..dimension).map(move |i| vector((0..dimension).map(|j| entry(matrix.clone(), i, j))))
    }));
    let step = op(
        "get",
        [op("qr", [op("rowstack", [stacked])]), Expr::Text("R")],
    );
    let initial = op("fill", [num(0.0), block_shape]);
    get(
        Expr::Scan(Box::new(step), Box::new(initial), Box::new(blocks_data)),
        blocks - 1,
    )
}

pub(super) fn real_shape(ty: &Type) -> Option<Vec<usize>> {
    let Type::Array { shape, elem } = ty else {
        return None;
    };
    if !matches!(elem.as_ref(), Type::Scalar(ScalarType::Real)) {
        return None;
    }
    shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) if *n > 0 => Some(*n as usize),
            _ => None,
        })
        .collect()
}

fn finite_bound(set: &ValueSet) -> bool {
    match set {
        ValueSet::Integers | ValueSet::NonNegIntegers | ValueSet::PosIntegers => true,
        ValueSet::Interval(lo, hi) => [lo, hi]
            .iter()
            .all(|x| x.is_finite() && (2.0 * **x as f32).is_finite()),
        ValueSet::CartPow(cell, _) => finite_bound(cell),
        _ => false,
    }
}

/// Infinities in a multivariate primitive can interact with covariance entries.
/// A few scalar flags cannot reconstruct those IEEE outcomes after compression.
pub(super) fn finite_data(m: &Module, root: NodeId, reducer: &Reducer) -> bool {
    let id = resolve(m, root, &reducer.boundary);
    match m.node(id) {
        Node::Ref(r) if r.ns == RefNs::SelfMod && reducer.boundary.contains(&r.name) => {
            let Some(bid) = m.binding_by_name(r.name) else {
                return false;
            };
            let rhs = m.binding(bid).rhs;
            ["external", "load_data", "elementof"]
                .iter()
                .any(|head| call(m, rhs, head).is_some())
                && m.valueset_of(root).is_some_and(finite_bound)
        }
        Node::Lit(Scalar::Real(x)) => x.is_finite() && (2.0 * *x as f32).is_finite(),
        Node::Lit(Scalar::Int(_)) => true,
        Node::Call(c) if matches!(c.head, flatppl_core::CallHead::Builtin(h) if matches!(m.resolve(h), "vector" | "rowstack" | "colstack" | "cat" | "fill" | "neg")) => {
            c.args.iter().all(|&child| finite_data(m, child, reducer))
        }
        _ => m.valueset_of(root).is_some_and(finite_bound),
    }
}

pub(super) fn reduce(reducer: &mut Reducer, m: &mut Module, id: NodeId) -> Option<Expr> {
    let factor = per_cell(m, id, &reducer.boundary).or_else(|| {
        super::per_cell_with_invariants(m, id, &reducer.boundary)
            .filter(|f| matches!(m.resolve(f.constructor), "Multinomial" | "InverseWishart"))
    })?;
    let family = m.resolve(factor.constructor).to_string();
    if !matches!(
        family.as_str(),
        "MvNormal" | "Dirichlet" | "Wishart" | "InverseWishart" | "Multinomial"
    ) || !data_only(m, factor.observed, &reducer.allowed, &reducer.boundary)
    {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(factor.observed)? else {
        return None;
    };
    let [Dim::Static(count)] = shape.as_ref() else {
        return None;
    };
    if *count < 2 {
        return None;
    }
    let count = *count as usize;
    let event = if family == "Multinomial" {
        let Type::Array { shape, elem } = elem.as_ref() else {
            return None;
        };
        if !matches!(elem.as_ref(), Type::Scalar(ScalarType::Integer)) {
            return None;
        }
        shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(n) if *n > 0 => Some(*n as usize),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?
    } else {
        real_shape(elem)?
    };
    let d = *event.first()?;
    let matrix = matches!(family.as_str(), "Wishart" | "InverseWishart");
    if (matrix && event != [d, d]) || (!matrix && event != [d]) {
        return None;
    }
    let names = flatppl_infer::distribution_param_names(&family)?;
    let ranks = flatppl_infer::distribution_param_ranks(&family)?;
    let parameters = factor
        .parameters
        .iter()
        .map(|&(name, value)| {
            let index = names.iter().position(|p| p == m.resolve(name))?;
            let value = if ranks[index] == 0 {
                reducer.scalar(m, value)?
            } else {
                // The product broadcast holds an entire vector/matrix in one cell.
                let wrapped = call(m, resolve(m, value, &reducer.boundary), "vector")?;
                let [only] = *wrapped.args else { return None };
                if real_shape(m.type_of(only)?)?.len() != ranks[index] {
                    return None;
                }
                only
            };
            Some((name, value))
        })
        .collect::<Option<Vec<_>>>()?;
    if parameters.len() != names.len() {
        return None;
    }
    let parameter = |name: &str| {
        parameters
            .iter()
            .find(|(s, _)| m.resolve(*s) == name)
            .map(|(_, id)| Expr::Node(*id))
    };
    let mu = parameter("mu");
    let cov = parameter("cov");
    let alpha = parameter("alpha");
    let nu = parameter("nu");
    let scale = parameter("scale");
    let observed = factor.observed;
    let data = Expr::Node(observed);
    let n = num(count as f64);
    match family.as_str() {
        "Multinomial" => {
            let trials_id = parameters
                .iter()
                .find(|(name, _)| m.resolve(*name) == "n")?
                .1;
            let trials = resolve(m, trials_id, &reducer.boundary);
            // The trial count fixes the support. Large factorial differences
            // retain primitive scoring, as in the scalar count reductions.
            if !matches!(m.node(trials), Node::Lit(Scalar::Int(n)) if *n > 0 && *n < 64) {
                return None;
            }
            let Node::Lit(Scalar::Int(trial_count)) = m.node(trials) else {
                return None;
            };
            if count > (1 << 24) || count.checked_mul(*trial_count as usize)? > i32::MAX as usize {
                return None;
            }
            let trials = Expr::Node(trials);
            let probs = parameter("p")?;
            let valid = |row: Expr| {
                op(
                    "land",
                    [
                        op("lall", [super::map("ge", [row.clone(), Expr::Integer(0)])]),
                        op(
                            "iszero",
                            [op(
                                "sub",
                                [
                                    op("sum", [super::map("real", [row])]),
                                    op("real", [trials.clone()]),
                                ],
                            )],
                        ),
                    ],
                )
            };
            let key = [observed, trials_id];
            let flags = shared(m, cells(valid(Expr::Local), data.clone()));
            let first = op(
                "integer",
                [op(
                    "round",
                    [op(
                        "minimum",
                        [super::map(
                            "real",
                            [super::map(
                                "ifelse",
                                [
                                    flags.clone(),
                                    op("indicesof", [flags.clone()]),
                                    Expr::Integer(count as i64),
                                ],
                            )],
                        )],
                    )],
                )],
            );
            let first = op("get", [data.clone(), first]);
            let center = reducer.summary_for(
                m,
                &key,
                "multinomial_center",
                op(
                    "ifelse",
                    [
                        op("lany", [flags.clone()]),
                        first,
                        op("fill", [Expr::Integer(0), Expr::Integer(d as i64)]),
                    ],
                ),
            );
            let valid_count = reducer.summary_for(
                m,
                &key,
                "multinomial_valid_count",
                op(
                    "sum",
                    [super::map(
                        "ifelse",
                        [flags, Expr::Integer(1), Expr::Integer(0)],
                    )],
                ),
            );
            let counts = reducer.summary_for(
                m,
                &key,
                "multinomial_counts",
                vector((0..d).map(|j| {
                    op(
                        "sum",
                        [cells(
                            op(
                                "ifelse",
                                [valid(Expr::Local), get(Expr::Local, j), Expr::Integer(0)],
                            ),
                            data.clone(),
                        )],
                    )
                })),
            );
            let factorials = reducer.summary_for(
                m,
                &key,
                "multinomial_relative_factorials",
                op(
                    "sum",
                    [held_cells(
                        op(
                            "ifelse",
                            [
                                valid(Expr::Local),
                                op(
                                    "sum",
                                    [super::map(
                                        "sub",
                                        [
                                            super::map(
                                                "loggamma",
                                                [super::map("add", [Expr::Local, num(1.0)])],
                                            ),
                                            super::map(
                                                "loggamma",
                                                [super::map("add", [Expr::Held(0), num(1.0)])],
                                            ),
                                        ],
                                    )],
                                ),
                                num(0.0),
                            ],
                        ),
                        data,
                        [center.clone()],
                    )],
                ),
            );
            let delta = super::map(
                "sub",
                [
                    counts,
                    super::map("mul", [valid_count.clone(), center.clone()]),
                ],
            );
            let skip = super::map(
                "lor",
                [
                    super::map("iszero", [delta.clone()]),
                    super::map(
                        "land",
                        [
                            super::map("iszero", [probs.clone()]),
                            super::map("gt", [center.clone(), Expr::Integer(0)]),
                        ],
                    ),
                ],
            );
            let safe_probs = super::map("ifelse", [skip, num(1.0), probs]);
            let correction = op(
                "sub",
                [dot(delta, super::map("log", [safe_probs])), factorials],
            );
            let anchor = reducer.anchor(m, &factor, &parameters, center);
            let normalizer = op(
                "ifelse",
                [
                    op("iszero", [valid_count.clone()]),
                    num(0.0),
                    op("mul", [valid_count.clone(), anchor]),
                ],
            );
            let penalty = op(
                "ifelse",
                [
                    op("lt", [valid_count.clone(), Expr::Integer(count as i64)]),
                    op("neg", [Expr::Constant("inf")]),
                    num(0.0),
                ],
            );
            let correction = op(
                "ifelse",
                [op("iszero", [valid_count]), num(0.0), correction],
            );
            Some(op("add", [op("add", [normalizer, correction]), penalty]))
        }
        "InverseWishart" => {
            if !finite_data(m, observed, reducer)
                || (!reducer.preparing
                    && !data_only(m, observed, &Default::default(), &reducer.boundary))
            {
                return None;
            }
            let nu = nu?;
            let scale = scale?;
            let identity = op(
                "rowstack",
                [vector((0..d).map(|j| {
                    vector((0..d).map(|k| num(if j == k { 1.0 } else { 0.0 })))
                }))],
            );
            let valid_factor = |l: Expr| {
                op(
                    "land",
                    [
                        op("lall", [super::map("isfinite", [l.clone()])]),
                        op(
                            "lall",
                            [vector(
                                (0..d).map(|j| op("gt", [entry(l.clone(), j, j), num(0.0)])),
                            )],
                        ),
                    ],
                )
            };
            let raw_li = op("lower_cholesky", [Expr::Local]);
            let all_valid = reducer.summary(
                m,
                observed,
                "inverse_wishart_valid",
                op("lall", [cells(valid_factor(raw_li.clone()), data.clone())]),
            );
            let first = get(data.clone(), 0);
            let first_l = op("lower_cholesky", [first.clone()]);
            let center = reducer.summary(
                m,
                observed,
                "inverse_wishart_center",
                op("ifelse", [valid_factor(first_l), first, identity.clone()]),
            );
            let lc = shared(m, op("lower_cholesky", [center]));
            // Generic solves must not turn an observation's failed Cholesky
            // into a new preparation error. Carry that failure into scoring.
            let li = op("ifelse", [valid_factor(raw_li.clone()), raw_li, identity]);
            let relative_det = op(
                "sum",
                [vector((0..d).map(|j| {
                    super::relative_log(entry(li.clone(), j, j), entry(Expr::Held(0), j, j))
                }))],
            );
            let log_delta = reducer.summary(
                m,
                observed,
                "inverse_wishart_logdet",
                op(
                    "mul",
                    [
                        num(2.0),
                        op(
                            "sum",
                            [held_cells(relative_det, data.clone(), [lc.clone()])],
                        ),
                    ],
                ),
            );
            let inverse = op(
                "colstack",
                [vector((0..d).map(|j| {
                    op(
                        "linsolve",
                        [
                            li.clone(),
                            vector((0..d).map(|k| num(if j == k { 1.0 } else { 0.0 }))),
                        ],
                    )
                }))],
            );
            let scales = reducer.summary(
                m,
                observed,
                "inverse_wishart_scales",
                vector((0..d).map(|j| {
                    op(
                        "maximum",
                        [cells(
                            op(
                                "maximum",
                                [super::map(
                                    "abs",
                                    [vector((0..d).map(|k| entry(inverse.clone(), k, j)))],
                                )],
                            ),
                            data.clone(),
                        )],
                    )
                })),
            );
            // Stack all inverse-factor rows. Normalize columns before QR,
            // then multiply by the scale Cholesky before any squaring.
            let row_count = count.checked_mul(d)?;
            let inverse_rows = op(
                "cat",
                (0..d).map(|j| {
                    cells(
                        vector((0..d).map(|k| entry(inverse.clone(), j, k))),
                        data.clone(),
                    )
                }),
            );
            let rows = held_cells(
                super::map(
                    "divide",
                    [
                        Expr::Local,
                        super::map("mul", [Expr::Held(0), op("sqrt", [n.clone()])]),
                    ],
                ),
                inverse_rows,
                [scales.clone()],
            );
            let r = reducer.summary(
                m,
                observed,
                "inverse_wishart_factor",
                residual_factor(rows, row_count, d),
            );
            let invalid = op("divide", [num(0.0), num(0.0)]);
            let validity = op("ifelse", [all_valid, num(1.0), invalid]);
            let r = super::map("mul", [r, validity.clone()]);
            let log_delta = op("mul", [log_delta, validity]);
            let ls = shared(m, op("lower_cholesky", [scale.clone()]));
            let scaled_l = op(
                "rowstack",
                [vector((0..d).map(|j| {
                    super::map(
                        "mul",
                        [
                            get(scales.clone(), j),
                            vector((0..d).map(|k| entry(ls.clone(), j, k))),
                        ],
                    )
                }))],
            );
            let product = shared(m, op("mul", [r, scaled_l]));
            let trace = op(
                "sum",
                [vector((0..d).flat_map(|j| {
                    let product = product.clone();
                    (0..d).map(move |k| {
                        op(
                            "mul",
                            [entry(product.clone(), j, k), entry(product.clone(), j, k)],
                        )
                    })
                }))],
            );
            let log_scale_delta = op(
                "mul",
                [
                    num(2.0),
                    op(
                        "sum",
                        [vector((0..d).map(|j| {
                            super::relative_log(entry(ls.clone(), j, j), entry(lc.clone(), j, j))
                        }))],
                    ),
                ],
            );
            let coefficient = op("mul", [num(0.5), op("add", [nu, num(d as f64 + 1.0)])]);
            let anchor = reducer.anchor(m, &factor, &parameters, scale);
            Some(op(
                "sub",
                [
                    op(
                        "sub",
                        [
                            op(
                                "mul",
                                [
                                    n.clone(),
                                    op(
                                        "add",
                                        [
                                            op("add", [anchor, num(d as f64 / 2.0)]),
                                            op("mul", [coefficient.clone(), log_scale_delta]),
                                        ],
                                    ),
                                ],
                            ),
                            op("mul", [coefficient, log_delta]),
                        ],
                    ),
                    op("mul", [op("mul", [num(0.5), n]), trace]),
                ],
            ))
        }
        "MvNormal" => {
            if !finite_data(m, observed, reducer) {
                return None;
            }
            let mu = mu?;
            let cov = cov?;
            let center_values = (0..d)
                .map(|j| {
                    let column = cells(get(Expr::Local, j), data.clone());
                    let mixed = op(
                        "land",
                        [
                            op("lt", [op("minimum", [column.clone()]), num(0.0)]),
                            op("gt", [op("maximum", [column.clone()]), num(0.0)]),
                        ],
                    );
                    let first = op("ifelse", [mixed, num(0.0), get(column.clone(), 0)]);
                    op(
                        "add",
                        [
                            first.clone(),
                            op(
                                "sum",
                                [super::map(
                                    "divide",
                                    [super::map("sub", [column, first]), n.clone()],
                                )],
                            ),
                        ],
                    )
                })
                .collect::<Vec<_>>();
            let center = reducer.summary(m, observed, "mv_center", vector(center_values));
            let residual = op("sub", [Expr::Local, Expr::Held(0)]);
            let scales = vector((0..d).map(|j| {
                op(
                    "maximum",
                    [held_cells(
                        op("abs", [get(residual.clone(), j)]),
                        data.clone(),
                        [center.clone()],
                    )],
                )
            }));
            let scales = reducer.summary(m, observed, "mv_scales", scales);
            let safe_scales = super::map(
                "ifelse",
                [
                    super::map("iszero", [Expr::Held(1)]),
                    num(1.0),
                    Expr::Held(1),
                ],
            );
            let scaled = super::map("divide", [residual, safe_scales]);
            let residual_mean = vector((0..d).map(|j| {
                op(
                    "sum",
                    [held_cells(
                        op("divide", [get(scaled.clone(), j), n.clone()]),
                        data.clone(),
                        [center.clone(), scales.clone()],
                    )],
                )
            }));
            let residual_mean = reducer.summary(m, observed, "mv_residual_mean", residual_mean);
            let rows = held_cells(
                super::map("divide", [scaled, op("sqrt", [n.clone()])]),
                data,
                [center.clone(), scales.clone()],
            );
            let factor_r = reducer.summary(
                m,
                observed,
                "mv_residual_factor",
                residual_factor(rows, count, d),
            );
            let anchor = reducer.anchor(m, &factor, &parameters, center.clone());
            let anchor = shared(m, anchor);
            let finite = op("isfinite", [anchor.clone()]);
            let l = shared(m, op("lower_cholesky", [cov]));
            let spread = vector((0..d).map(|i| {
                let row = vector((0..d).map(|j| entry(factor_r.clone(), i, j)));
                let rhs = super::map("mul", [row, scales.clone()]);
                let rhs = super::map("ifelse", [finite.clone(), rhs, num(0.0)]);
                let whitened = op("linsolve", [l.clone(), rhs]);
                dot(whitened.clone(), whitened)
            }));
            let spread = op("sum", [spread]);
            let nonzero = op(
                "lany",
                [super::map(
                    "lnot",
                    [super::map("iszero", [residual_mean.clone()])],
                )],
            );
            let active = op("land", [finite.clone(), nonzero]);
            let delta = super::map(
                "ifelse",
                [active.clone(), op("sub", [center, mu]), num(0.0)],
            );
            let mean = super::map("mul", [residual_mean, scales]);
            let cross = dot(
                op("linsolve", [l.clone(), delta]),
                op("linsolve", [l, mean]),
            );
            let cross = op("ifelse", [active, cross, num(0.0)]);
            let corrected = op(
                "sub",
                [
                    anchor.clone(),
                    op("add", [cross, op("mul", [num(0.5), spread])]),
                ],
            );
            Some(op("mul", [n, op("ifelse", [finite, corrected, anchor])]))
        }
        "Dirichlet" => {
            let alpha = alpha?;
            // The primitive retains the normalizer at an interior simplex point.
            let support = op("stdsimplex", [Expr::Integer(d as i64)]);
            let first = get(data.clone(), 0);
            let valid = super::map(
                "land",
                [
                    super::map("isfinite", [first.clone()]),
                    // A portable center must survive float32 subnormal flush.
                    super::map("ge", [first.clone(), num(f64::from(f32::MIN_POSITIVE))]),
                ],
            );
            let center = op(
                "ifelse",
                [
                    op(
                        "land",
                        [
                            op("lall", [valid]),
                            op("in", [first.clone(), support.clone()]),
                        ],
                    ),
                    first,
                    op("fill", [num(1.0 / d as f64), Expr::Integer(d as i64)]),
                ],
            );
            let center = reducer.summary(m, observed, "dirichlet_center", center);
            let safe_point = op(
                "ifelse",
                [
                    op("in", [Expr::Local, support.clone()]),
                    Expr::Local,
                    Expr::Held(0),
                ],
            );
            let logs = vector((0..d).map(|j| {
                let coordinate = get(safe_point.clone(), j);
                let safe = op(
                    "ifelse",
                    [op("iszero", [coordinate.clone()]), num(1.0), coordinate],
                );
                op(
                    "sum",
                    [held_cells(
                        super::relative_log(safe, get(Expr::Held(0), j)),
                        data.clone(),
                        [center.clone()],
                    )],
                )
            }));
            let logs = reducer.summary(m, observed, "dirichlet_finite_logs", logs);
            let zeros = vector((0..d).map(|j| {
                op(
                    "sum",
                    [held_cells(
                        op(
                            "ifelse",
                            [
                                op("iszero", [get(safe_point.clone(), j)]),
                                Expr::Integer(1),
                                Expr::Integer(0),
                            ],
                        ),
                        data.clone(),
                        [center.clone()],
                    )],
                )
            }));
            let zeros = reducer.summary(m, observed, "dirichlet_zeros", zeros);
            let coefficients = super::map("sub", [alpha, num(1.0)]);
            let infinite = super::map(
                "land",
                [
                    super::map("lnot", [super::map("iszero", [coefficients.clone()])]),
                    super::map("gt", [zeros, Expr::Integer(0)]),
                ],
            );
            let logs = super::map(
                "ifelse",
                [infinite, op("neg", [Expr::Constant("inf")]), logs],
            );
            let correction = dot(coefficients, logs);
            let anchor = reducer.anchor(m, &factor, &parameters, center);
            let valid_count = reducer.summary(
                m,
                observed,
                "dirichlet_valid_count",
                op(
                    "sum",
                    [cells(
                        op(
                            "ifelse",
                            [
                                op("in", [Expr::Local, support]),
                                Expr::Integer(1),
                                Expr::Integer(0),
                            ],
                        ),
                        Expr::Node(observed),
                    )],
                ),
            );
            let normalizer = op(
                "ifelse",
                [
                    op("iszero", [valid_count.clone()]),
                    num(0.0),
                    op("mul", [valid_count.clone(), anchor]),
                ],
            );
            let correction = op(
                "ifelse",
                [op("iszero", [valid_count.clone()]), num(0.0), correction],
            );
            let penalty = op(
                "ifelse",
                [
                    op("lt", [valid_count, Expr::Integer(count as i64)]),
                    op("neg", [Expr::Constant("inf")]),
                    num(0.0),
                ],
            );
            Some(op("add", [op("add", [normalizer, correction]), penalty]))
        }
        "Wishart" => {
            if !finite_data(m, observed, reducer) {
                return None;
            }
            let nu = nu?;
            let scale = scale?;
            let center = reducer.summary(m, observed, "wishart_center", get(data.clone(), 0));
            let lc = shared(m, op("lower_cholesky", [center.clone()]));
            let center_gram = shared(m, op("mul", [lc.clone(), op("transpose", [lc.clone()])]));
            // Keep matrix intermediates inside a scalar-valued map, so backends
            // need not reconstruct nested matrix cells from a mapped result.
            let cholesky = op("lower_cholesky", [Expr::Local]);
            let log_delta = (0..d).map(|j| {
                op(
                    "sum",
                    [held_cells(
                        super::relative_log(
                            entry(cholesky.clone(), j, j),
                            entry(Expr::Held(0), j, j),
                        ),
                        data.clone(),
                        [lc.clone()],
                    )],
                )
            });
            let log_delta = reducer.summary(
                m,
                observed,
                "wishart_relative_logdet",
                op("mul", [num(2.0), op("sum", [vector(log_delta)])]),
            );
            let gram_delta = vector((0..d).map(|j| {
                vector((0..d).map(|k| {
                    let delta = op(
                        "sub",
                        [
                            entry(
                                op(
                                    "mul",
                                    [cholesky.clone(), op("transpose", [cholesky.clone()])],
                                ),
                                j,
                                k,
                            ),
                            entry(Expr::Held(0), j, k),
                        ],
                    );
                    op(
                        "sum",
                        [held_cells(
                            op("divide", [delta, n.clone()]),
                            data.clone(),
                            [center_gram.clone()],
                        )],
                    )
                }))
            }));
            let gram_delta = reducer.summary(
                m,
                observed,
                "wishart_mean_delta",
                op("rowstack", [gram_delta]),
            );
            let scale_l = shared(m, op("lower_cholesky", [scale]));
            let trace = (0..d).map(|j| {
                let column = vector((0..d).map(|k| entry(gram_delta.clone(), k, j)));
                get(
                    op(
                        "linsolve",
                        [
                            op("transpose", [scale_l.clone()]),
                            op("linsolve", [scale_l.clone(), column]),
                        ],
                    ),
                    j,
                )
            });
            let coefficient = op("mul", [num(0.5), op("sub", [nu, num(d as f64 + 1.0)])]);
            let anchor = reducer.anchor(m, &factor, &parameters, center);
            Some(op(
                "sub",
                [
                    op(
                        "add",
                        [
                            op("mul", [n.clone(), anchor]),
                            op("mul", [coefficient, log_delta]),
                        ],
                    ),
                    op(
                        "mul",
                        [op("mul", [num(0.5), n]), op("sum", [vector(trace)])],
                    ),
                ],
            ))
        }
        _ => unreachable!(),
    }
}
