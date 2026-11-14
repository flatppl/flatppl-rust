//! Correlation-matrix likelihoods retain their primitive's support predicate.

use super::multivariate::{cells, entry, real_shape, shared, vector};
use super::{Expr, Reducer, data_only, materialize, num, op, per_cell_with_invariants, resolve};
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, Type};

pub(super) fn reduce(reducer: &mut Reducer, m: &mut Module, id: NodeId) -> Option<Expr> {
    let factor = per_cell_with_invariants(m, id, &reducer.boundary)?;
    let family = m.resolve(factor.constructor).to_owned();
    if !matches!(family.as_str(), "LKJ" | "LKJCholesky")
        || !data_only(m, factor.observed, &reducer.allowed, &reducer.boundary)
        || (!reducer.preparing
            && !data_only(m, factor.observed, &Default::default(), &reducer.boundary))
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
    let event = real_shape(elem)?;
    let [dimension, columns] = event.as_slice() else {
        return None;
    };
    if dimension != columns {
        return None;
    }
    let dimension = *dimension;
    let parameters = factor
        .parameters
        .iter()
        .map(|&(name, value)| Some((name, reducer.scalar(m, value)?)))
        .collect::<Option<Vec<_>>>()?;
    if parameters.len() != 2 {
        return None;
    }
    let n_id = parameters.iter().find(|(s, _)| m.resolve(*s) == "n")?.1;
    let eta_id = parameters.iter().find(|(s, _)| m.resolve(*s) == "eta")?.1;
    let n_id = resolve(m, n_id, &reducer.boundary);
    if !matches!(m.node(n_id), Node::Lit(Scalar::Int(n)) if *n > 0 && usize::try_from(*n).ok() == Some(dimension))
    {
        return None;
    }
    let identity = op(
        "rowstack",
        [vector((0..dimension).map(|row| {
            vector((0..dimension).map(|col| num(if row == col { 1.0 } else { 0.0 })))
        }))],
    );
    let reference_eta = materialize(m, num(1.0));
    let reference_parameters = parameters
        .iter()
        .map(|&(name, value)| {
            (
                name,
                if m.resolve(name) == "eta" {
                    reference_eta
                } else {
                    value
                },
            )
        })
        .collect::<Vec<_>>();
    let reference = reducer.anchor(m, &factor, &reference_parameters, Expr::Local);
    let flags = shared(
        m,
        cells(op("isfinite", [reference]), Expr::Node(factor.observed)),
    );
    let key = [factor.observed, n_id];
    let (count_kind, logs_kind) = if family == "LKJ" {
        ("lkj_valid_count", "lkj_log_diagonals")
    } else {
        ("lkj_cholesky_valid_count", "lkj_cholesky_log_diagonals")
    };
    let valid_count = reducer.summary_for(
        m,
        &key,
        count_kind,
        op(
            "sum",
            [super::map("ifelse", [flags.clone(), num(1.0), num(0.0)])],
        ),
    );
    // Membership is data-only. Sanitize the entire cell before Cholesky/log,
    // so invalid observations contribute only the support penalty.
    let reference = reducer.anchor(m, &factor, &reference_parameters, Expr::Local);
    let safe_cell = op(
        "ifelse",
        [op("isfinite", [reference]), Expr::Local, identity.clone()],
    );
    let log_cell = if family == "LKJ" {
        op("lower_cholesky", [safe_cell])
    } else {
        safe_cell
    };
    // Factor each observation once, outside the per-diagonal maps.
    let log_cells = shared(m, cells(log_cell, Expr::Node(factor.observed)));
    let logs = if dimension == 1 {
        num(0.0)
    } else {
        reducer.summary_for(
            m,
            &key,
            logs_kind,
            vector((1..dimension).map(|j| {
                op(
                    "sum",
                    [cells(
                        op("log", [entry(Expr::Local, j, j)]),
                        log_cells.clone(),
                    )],
                )
            })),
        )
    };
    let any_valid = op("gt", [valid_count.clone(), num(0.0)]);
    let safe_eta = materialize(
        m,
        op("ifelse", [any_valid.clone(), Expr::Node(eta_id), num(1.0)]),
    );
    // Keep the original parameter domain check even when every row is invalid.
    // An off-support point also keeps inactive nonfinite eta adjoints harmless.
    let zero = super::map("mul", [num(0.0), identity.clone()]);
    let point = op("ifelse", [any_valid.clone(), identity, zero]);
    let anchor = reducer.anchor(m, &factor, &parameters, point);
    let normalizer = op(
        "mul",
        [
            valid_count.clone(),
            op("ifelse", [any_valid, anchor, num(0.0)]),
        ],
    );
    let twice_eta = op("mul", [num(2.0), Expr::Node(safe_eta)]);
    let correction = if dimension == 1 {
        num(0.0)
    } else {
        op(
            "sum",
            [vector((1..dimension).map(|j| {
                let shift = if family == "LKJ" {
                    -2.0
                } else {
                    (dimension - j) as f64 - 3.0
                };
                op(
                    "mul",
                    [
                        op("add", [twice_eta.clone(), num(shift)]),
                        super::multivariate::get(logs.clone(), j - 1),
                    ],
                )
            }))],
        )
    };
    let penalty = op(
        "ifelse",
        [
            op("lt", [valid_count, num(count as f64)]),
            op("neg", [Expr::Constant("inf")]),
            num(0.0),
        ],
    );
    Some(op("add", [op("add", [normalizer, correction]), penalty]))
}
