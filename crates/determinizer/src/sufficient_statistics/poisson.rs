//! A data-centered primitive anchor retains accurate large-count scoring.

use super::{Expr, Factor, Reducer, map, materialize, num, op, relative_log};
use flatppl_core::{Dim, Module, NodeId, ScalarType, Type};

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, f: &Factor) -> Option<NodeId> {
    if m.resolve(f.constructor) != "Poisson"
        || (!r.preparing && !r.fixed(m, f.observed))
        || !r.data_only(m, f.observed)
    {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(f.observed)? else {
        return None;
    };
    let [Dim::Static(count)] = shape.as_ref() else {
        return None;
    };
    if *count < 2 || *count > 1 << 24 || !matches!(elem.as_ref(), Type::Scalar(ScalarType::Integer))
    {
        return None;
    }
    let count = *count;
    let n = num(count as f64);
    let parameter = f
        .parameters
        .iter()
        .find(|(name, _)| m.resolve(*name) == "rate")?;
    let rate = r.scalar(m, parameter.1)?;
    let parameters = [(parameter.0, rate)];
    let observed = f.observed;
    let x = Expr::Node(observed);
    let flags = map("ge", [x.clone(), Expr::Integer(0)]);
    let valid = r.summary(
        m,
        observed,
        "poisson_count_valid",
        op("lall", [flags.clone()]),
    );
    let safe = Expr::Node(materialize(m, map("ifelse", [flags, x, Expr::Integer(0)])));
    let real = map("real", [safe.clone()]);
    let reference = r.summary(
        m,
        observed,
        "poisson_reference_rate",
        op("max", [op("mean", [real.clone()]), num(1.0)]),
    );
    let distance = map("abs", [map("sub", [real.clone(), reference.clone()])]);
    let nearest = map("le", [distance.clone(), op("minimum", [distance])]);
    let index = op(
        "integer",
        [op(
            "round",
            [op(
                "minimum",
                [map(
                    "real",
                    [map(
                        "ifelse",
                        [
                            nearest.clone(),
                            op("indicesof", [nearest]),
                            Expr::Integer(count as i64),
                        ],
                    )],
                )],
            )],
        )],
    );
    let center = r.summary(
        m,
        observed,
        "poisson_count_center",
        op("get", [safe.clone(), index]),
    );
    let reference_id = materialize(m, reference.clone());
    let reference_parameters = [(parameter.0, reference_id)];
    let held = materialize(m, Expr::Held(0));
    let body = r.anchor(m, f, &[(parameter.0, held)], Expr::Local);
    let scores = super::multivariate::held_cells(body, safe, [reference]);
    let anchor = r.anchor(m, f, &reference_parameters, center.clone());
    let correction = r.summary(
        m,
        observed,
        "poisson_reference_residual",
        op("sum", [map("sub", [scores, anchor])]),
    );
    let residual = r.summary(
        m,
        observed,
        "poisson_count_residual",
        op("sum", [map("sub", [real, op("real", [center.clone()])])]),
    );
    let anchor = r.anchor(m, f, &parameters, center);
    let anchor = Expr::Node(materialize(m, anchor));
    let finite = op("isfinite", [anchor.clone()]);
    let active = op(
        "land",
        [
            finite.clone(),
            op("lnot", [op("iszero", [residual.clone()])]),
        ],
    );
    let safe_rate = op(
        "ifelse",
        [active, Expr::Node(rate), Expr::Node(reference_id)],
    );
    let slope = relative_log(safe_rate, Expr::Node(reference_id));
    let base = op("mul", [n, anchor]);
    let corrected = op(
        "add",
        [
            base.clone(),
            op("add", [correction, op("mul", [residual, slope])]),
        ],
    );
    let score = op("ifelse", [finite, corrected, base]);
    r.count += 1;
    Some(materialize(
        m,
        op("ifelse", [valid, score, op("neg", [Expr::Constant("inf")])]),
    ))
}
