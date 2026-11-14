//! Anchored scalar families whose statistics also depend on fixed parameters.

use super::{
    Expr, Reducer, data_only, fixed_values, map, materialize, num, op, per_cell_with_invariants,
    positive_in_both_precisions, relative_log,
};
use flatppl_core::{Module, NodeId, ScalarType, Type, ValueSet};

fn positive(m: &Module, id: NodeId, reducer: &Reducer) -> bool {
    if fixed_values(m, id, &reducer.boundary)
        .is_some_and(|xs| xs.iter().all(|x| positive_in_both_precisions(*x)))
    {
        return true;
    }
    let set = match m.valueset_of(id) {
        Some(ValueSet::CartPow(set, _)) => set.as_ref(),
        Some(set) => set,
        None => return false,
    };
    matches!(set, ValueSet::Interval(lo, hi) if positive_in_both_precisions(*lo) && positive_in_both_precisions(*hi))
}

fn shared(m: &mut Module, value: Expr) -> Expr {
    Expr::Node(materialize(m, value))
}

fn bounded_center(center: Expr, minimum: Expr, maximum: Expr) -> Expr {
    op("max", [minimum, op("min", [maximum, center])])
}

/// Inactive corrections must stay finite in reverse mode, including 0 * inf.
fn active_value(active: Expr, value: Expr, fallback: f64) -> Expr {
    op("ifelse", [active, value, num(fallback)])
}

pub(super) fn reduce(reducer: &mut Reducer, m: &mut Module, id: NodeId) -> Option<Expr> {
    let f = per_cell_with_invariants(m, id, &reducer.boundary)?;
    let name = m.resolve(f.constructor).to_owned();
    if !matches!(
        name.as_str(),
        "ChiSquared"
            | "InverseGamma"
            | "Pareto"
            | "Weibull"
            | "NegativeBinomial"
            | "NegativeBinomial2"
    ) || !data_only(m, f.observed, &reducer.allowed, &reducer.boundary)
        || (!reducer.preparing && !data_only(m, f.observed, &Default::default(), &reducer.boundary))
    {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(f.observed)?.clone() else {
        return None;
    };
    let [flatppl_core::Dim::Static(count)] = shape.as_ref() else {
        return None;
    };
    if *count < 2 {
        return None;
    }
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, value)| Some((name, reducer.scalar(m, value)?)))
        .collect::<Option<Vec<_>>>()?;
    let param = |key: &str| {
        parameters
            .iter()
            .find(|(name, _)| m.resolve(*name) == key)
            .map(|(_, value)| *value)
    };
    let shape = param(if name == "ChiSquared" { "k" } else { "shape" });
    let scale = param("scale");
    let dispersion = param(if name == "NegativeBinomial" {
        "alpha"
    } else {
        "psi"
    });
    let beta = param("beta");
    let mu = param("mu");
    let observed = f.observed;
    let x = Expr::Node(observed);
    let n = num(*count as f64);
    match name.as_str() {
        "ChiSquared" | "InverseGamma" | "Pareto" | "Weibull"
            if matches!(*elem, Type::Scalar(ScalarType::Real))
                && positive(m, observed, reducer) =>
        {
            let shape = shape?;
            let scale = if name == "ChiSquared" {
                num(0.5)
            } else {
                Expr::Node(scale?)
            };
            if name == "Weibull"
                && (!data_only(m, shape, &reducer.allowed, &reducer.boundary)
                    || !positive(m, shape, reducer)
                    || (!reducer.preparing
                        && !data_only(m, shape, &Default::default(), &reducer.boundary)))
            {
                return None;
            }
            let k = if name == "ChiSquared" {
                op("divide", [Expr::Node(shape), num(2.0)])
            } else {
                Expr::Node(shape)
            };
            let valid = map(
                "land",
                [
                    map("isfinite", [x.clone()]),
                    map("gt", [x.clone(), num(0.0)]),
                ],
            );
            let all_valid = reducer.summary(
                m,
                observed,
                "positive_finite_valid",
                op("lall", [valid.clone()]),
            );
            let maximum_valid = reducer.summary(
                m,
                observed,
                "positive_finite_maximum",
                op(
                    "maximum",
                    [map("ifelse", [valid.clone(), x.clone(), num(0.0)])],
                ),
            );
            let x = shared(m, map("ifelse", [valid, x, num(1.0)]));
            let keys = if name == "Weibull" {
                vec![observed, shape]
            } else {
                vec![observed]
            };
            let minimum = reducer.summary(
                m,
                observed,
                "positive_finite_minimum",
                op("minimum", [x.clone()]),
            );
            let center = if name == "Pareto" {
                minimum
            } else {
                let maximum = op("maximum", [x.clone()]);
                let value = if name == "ChiSquared" {
                    let first = op("get", [x.clone(), Expr::Integer(1)]);
                    op(
                        "add",
                        [
                            first.clone(),
                            op(
                                "sum",
                                [map("divide", [map("sub", [x.clone(), first]), n.clone()])],
                            ),
                        ],
                    )
                } else if name == "InverseGamma" {
                    // Every normalized reciprocal is <= 1, and their mean >= 1/n.
                    op(
                        "divide",
                        [
                            minimum.clone(),
                            op("mean", [map("divide", [minimum.clone(), x.clone()])]),
                        ],
                    )
                } else {
                    // Take logs before scaling: x/max can underflow for small k.
                    let powers = map(
                        "exp",
                        [map(
                            "mul",
                            [k.clone(), relative_log(x.clone(), maximum.clone())],
                        )],
                    );
                    op(
                        "mul",
                        [
                            maximum.clone(),
                            op(
                                "exp",
                                [op("divide", [op("log", [op("mean", [powers])]), k.clone()])],
                            ),
                        ],
                    )
                };
                reducer.summary_for(
                    m,
                    &keys,
                    if name == "ChiSquared" {
                        "chi_squared_center"
                    } else if name == "InverseGamma" {
                        "harmonic_center"
                    } else {
                        "weibull_center"
                    },
                    bounded_center(value, minimum, maximum),
                )
            };
            let logs = reducer.summary_for(
                m,
                &keys,
                if name == "ChiSquared" {
                    "chi_squared_relative_logs"
                } else if name == "Pareto" {
                    "pareto_relative_logs"
                } else if name == "Weibull" {
                    "weibull_relative_logs"
                } else {
                    "inverse_gamma_relative_logs"
                },
                op("sum", [relative_log(x.clone(), center.clone())]),
            );
            let anchor = reducer.anchor(m, &f, &parameters, center.clone());
            let anchor = shared(m, anchor);
            let finite = op("isfinite", [anchor.clone()]);
            let safe_k = active_value(finite.clone(), k.clone(), 1.0);
            let log_term = op(
                "mul",
                [
                    op(
                        if matches!(name.as_str(), "Weibull" | "ChiSquared") {
                            "sub"
                        } else {
                            "add"
                        },
                        [safe_k.clone(), num(1.0)],
                    ),
                    logs,
                ],
            );
            let correction = if name == "Pareto" {
                op("neg", [log_term])
            } else {
                let residual = reducer.summary_for(
                    m,
                    &keys,
                    if name == "ChiSquared" {
                        "chi_squared_residual"
                    } else if name == "Weibull" {
                        "weibull_power_residual"
                    } else {
                        "inverse_gamma_reciprocal_residual"
                    },
                    op(
                        "sum",
                        [if name == "ChiSquared" {
                            map("sub", [x.clone(), center.clone()])
                        } else if name == "Weibull" {
                            map(
                                "expm1",
                                [map("mul", [k, relative_log(x.clone(), center.clone())])],
                            )
                        } else {
                            map(
                                "divide",
                                [map("sub", [center.clone(), x.clone()]), x.clone()],
                            )
                        }],
                    ),
                );
                let active = op(
                    "land",
                    [
                        finite.clone(),
                        op("lnot", [op("iszero", [residual.clone()])]),
                    ],
                );
                let multiplier = if name == "ChiSquared" {
                    num(0.5)
                } else if name == "Weibull" {
                    op(
                        "pow",
                        [
                            op(
                                "divide",
                                [
                                    active_value(active.clone(), center, 1.0),
                                    active_value(active.clone(), scale, 1.0),
                                ],
                            ),
                            safe_k,
                        ],
                    )
                } else {
                    op(
                        "divide",
                        [
                            active_value(active.clone(), scale, 0.0),
                            active_value(active, center, 1.0),
                        ],
                    )
                };
                let power_term = op("mul", [multiplier, residual]);
                if matches!(name.as_str(), "Weibull" | "ChiSquared") {
                    op("sub", [log_term, power_term])
                } else {
                    op("neg", [op("add", [log_term, power_term])])
                }
            };
            let base = op("mul", [n, anchor]);
            let score = op(
                "ifelse",
                [finite, op("add", [base.clone(), correction]), base],
            );
            let invalid = op(
                "add",
                [
                    reducer.anchor(m, &f, &parameters, maximum_valid),
                    op("neg", [Expr::Constant("inf")]),
                ],
            );
            Some(op("ifelse", [all_valid, score, invalid]))
        }
        "NegativeBinomial" | "NegativeBinomial2"
            if matches!(*elem, Type::Scalar(ScalarType::Integer)) =>
        {
            let dispersion = dispersion?;
            if !data_only(m, dispersion, &reducer.allowed, &reducer.boundary)
                || !positive(m, dispersion, reducer)
                || (!reducer.preparing
                    && !data_only(m, dispersion, &Default::default(), &reducer.boundary))
            {
                return None;
            }
            // Concentrated, large-count float32 primitives lose low bits in
            // their normalizer. Moving that primitive can amplify the error.
            let small_dispersion =
                matches!(m.valueset_of(dispersion), Some(ValueSet::Interval(_, hi)) if *hi <= 64.0);
            let small_counts = fixed_values(m, observed, &reducer.boundary)
                .is_some_and(|xs| xs.iter().all(|x| *x >= 0.0 && *x < 64.0));
            if !small_dispersion && !small_counts {
                return None;
            }
            let nonnegative = fixed_values(m, observed, &reducer.boundary)
                .is_some_and(|xs| xs.iter().all(|x| *x >= 0.0))
                || matches!(m.valueset_of(observed), Some(ValueSet::CartPow(set, _)) if matches!(set.as_ref(), ValueSet::NonNegIntegers | ValueSet::PosIntegers) || matches!(set.as_ref(), ValueSet::Interval(lo, hi) if *lo >= 0.0 && hi.is_finite()));
            if !nonnegative {
                return None;
            }
            let a = Expr::Node(dispersion);
            let real_x = map("real", [x.clone()]);
            let mean = op("max", [op("mean", [real_x.clone()]), num(1.0)]);
            let rate = op("divide", [a.clone(), mean.clone()]);
            // Clip the reference only, so both ratios stay normal in float32.
            // Falling back to rate 1 would cancel huge counts against tiny rates.
            let safe_rate = op(
                "land",
                [
                    op("ge", [rate.clone(), num(1e-37)]),
                    op("le", [rate.clone(), num(1e37)]),
                ],
            );
            let reference_rate = bounded_center(rate, num(1e-37), num(1e37));
            let reference_value = if name == "NegativeBinomial" {
                reference_rate.clone()
            } else {
                op(
                    "ifelse",
                    [
                        safe_rate,
                        mean,
                        op("divide", [a.clone(), reference_rate.clone()]),
                    ],
                )
            };
            let varying = if name == "NegativeBinomial" {
                beta?
            } else {
                mu?
            };
            let reference_value = reducer.summary_for(
                m,
                &[observed, dispersion],
                if name == "NegativeBinomial" {
                    "negative_binomial_reference_rate"
                } else {
                    "negative_binomial2_reference_mean"
                },
                reference_value,
            );
            let reference_id = materialize(m, reference_value);
            let reference_parameters = parameters
                .iter()
                .map(|&(key, value)| {
                    (
                        key,
                        if m.resolve(key)
                            == if name == "NegativeBinomial" {
                                "beta"
                            } else {
                                "mu"
                            }
                        {
                            reference_id
                        } else {
                            value
                        },
                    )
                })
                .collect::<Vec<_>>();
            let center = reducer.summary(
                m,
                observed,
                "first_count",
                op("get", [x.clone(), Expr::Integer(1)]),
            );
            let residual = reducer.summary(
                m,
                observed,
                "count_residual",
                op(
                    "sum",
                    [map(
                        "sub",
                        [map("real", [x.clone()]), op("real", [center.clone()])],
                    )],
                ),
            );
            let reference_anchor = reducer.anchor(m, &f, &reference_parameters, center.clone());
            let held_dispersion = materialize(m, Expr::Held(0));
            let held_reference = materialize(m, Expr::Held(1));
            let held_parameters = parameters
                .iter()
                .map(|&(key, _)| {
                    (
                        key,
                        if m.resolve(key)
                            == if name == "NegativeBinomial" {
                                "alpha"
                            } else {
                                "psi"
                            }
                        {
                            held_dispersion
                        } else {
                            held_reference
                        },
                    )
                })
                .collect::<Vec<_>>();
            let reference_scores = super::multivariate::held_cells(
                reducer.anchor(m, &f, &held_parameters, Expr::Local),
                x,
                [Expr::Node(dispersion), Expr::Node(reference_id)],
            );
            let reference = reducer.summary_for(
                m,
                &[observed, dispersion],
                if name == "NegativeBinomial" {
                    "negative_binomial_reference_residual"
                } else {
                    "negative_binomial2_reference_residual"
                },
                op("sum", [map("sub", [reference_scores, reference_anchor])]),
            );
            let anchor = reducer.anchor(m, &f, &parameters, center);
            let anchor = shared(m, anchor);
            let finite = op("isfinite", [anchor.clone()]);
            let active = op(
                "land",
                [
                    finite.clone(),
                    op("lnot", [op("iszero", [residual.clone()])]),
                ],
            );
            let slope = if name == "NegativeBinomial" {
                op(
                    "log1p",
                    [active_value(active.clone(), Expr::Node(varying), 1.0)],
                )
            } else {
                op(
                    "log1p",
                    [op(
                        "divide",
                        [a, active_value(active.clone(), Expr::Node(varying), 1.0)],
                    )],
                )
            };
            let reference_slope = if name == "NegativeBinomial" {
                Expr::Node(reference_id)
            } else {
                op("divide", [Expr::Node(dispersion), Expr::Node(reference_id)])
            };
            let difference = active_value(
                active,
                op("sub", [slope, op("log1p", [reference_slope])]),
                0.0,
            );
            let base = op("mul", [n, anchor]);
            let score = op(
                "ifelse",
                [
                    finite,
                    op(
                        "add",
                        [
                            base.clone(),
                            op("sub", [reference, op("mul", [residual, difference])]),
                        ],
                    ),
                    base,
                ],
            );
            let valid = reducer.summary(
                m,
                observed,
                "nonnegative_count_valid",
                op("lall", [map("ge", [real_x, num(0.0)])]),
            );
            Some(op(
                "ifelse",
                [valid, score, op("neg", [Expr::Constant("inf")])],
            ))
        }
        _ => None,
    }
}
