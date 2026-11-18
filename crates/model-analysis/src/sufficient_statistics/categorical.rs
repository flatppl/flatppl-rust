//! Ordered categorical observations contribute one log probability per label.

use super::multivariate::{dot, shared, vector};
use super::{Expr, Factor, Reducer, call, fixed_values, map, materialize, num, op, resolve};
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, ScalarType, Symbol, Type};
use std::collections::HashSet;

mod materialized;
pub(super) use materialized::fold;

const MAX_CATEGORIES: u32 = 1 << 24;

fn fixed_counts(
    values: Vec<f64>,
    categories: u32,
    first: i64,
    integer: bool,
) -> Option<(Vec<i64>, bool)> {
    if categories as usize > values.len() {
        return None;
    }
    let index = |x: f64| {
        let i = x - first as f64;
        (x.fract() == 0.0 && i >= 0.0 && i < f64::from(categories)).then_some(i as usize)
    };
    let mut counts = vec![0; categories as usize];
    let mut valid = true;
    for x in values {
        let label = index(x);
        // A real close to an integer may become an admissible label in f32.
        if !integer && label != index(f64::from(x as f32)) {
            return None;
        }
        if let Some(label) = label {
            counts[label] += 1;
        } else {
            valid = false;
        }
    }
    Some((counts, valid))
}

/// Ignore invalid labels in the histogram and report their presence separately.
/// In particular, the final closed bin must not include the next category.
pub(crate) fn label_counts(
    m: &mut Module,
    observations: NodeId,
    integer: bool,
    categories: u32,
    one_based: bool,
    boundary: &HashSet<Symbol>,
) -> (NodeId, NodeId) {
    let first = i64::from(one_based);
    if let Some((counts, valid)) = fixed_values(m, observations, boundary)
        .and_then(|values| fixed_counts(values, categories, first, integer))
    {
        return (
            materialize(m, vector(counts.into_iter().map(Expr::Integer))),
            m.alloc(Node::Lit(Scalar::Bool(valid))),
        );
    }
    let y = Expr::Node(observations);
    let last = first + i64::from(categories) - 1;
    let mut valid = map(
        "land",
        [
            map("ge", [y.clone(), Expr::Integer(first)]),
            map("le", [y.clone(), Expr::Integer(last)]),
        ],
    );
    if !integer {
        valid = map(
            "land",
            [valid, map("le", [y.clone(), map("floor", [y.clone()])])],
        );
    }
    let valid = shared(m, valid);
    let below = if one_based {
        Expr::Integer(0)
    } else {
        op("neg", [Expr::Integer(1)])
    };
    let safe = map("ifelse", [valid.clone(), y, below]);
    let labels = if one_based {
        map("sub", [safe, Expr::Integer(1)])
    } else {
        safe
    };
    let edges = op(
        "linspace",
        [
            num(0.0),
            num(f64::from(categories)),
            Expr::Integer(i64::from(categories) + 1),
        ],
    );
    (
        materialize(m, op("bincounts", [edges, labels])),
        materialize(m, op("lall", [valid])),
    )
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, f: &Factor) -> Option<Expr> {
    let one_based = match m.resolve(f.constructor) {
        "Categorical" => true,
        "Categorical0" => false,
        _ => return None,
    };
    if !r.preparing && !r.fixed(m, f.observed) {
        return None;
    }
    let [(name, parameter)] = *f.parameters else {
        return None;
    };
    let wrapped = call(m, resolve(m, parameter, &r.boundary), "vector")?;
    let [probabilities] = *wrapped.args else {
        return None;
    };
    let Type::Array { shape, elem } = m.type_of(probabilities)? else {
        return None;
    };
    let [Dim::Static(categories)] = **shape else {
        return None;
    };
    if **elem != Type::Scalar(ScalarType::Real) {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(f.observed)? else {
        return None;
    };
    let [Dim::Static(rows)] = **shape else {
        return None;
    };
    let integer = match elem.as_ref() {
        Type::Scalar(ScalarType::Integer) => true,
        Type::Scalar(ScalarType::Real) => false,
        _ => return None,
    };
    // Keep the data ABI smaller and the edges and counts exact in both target precisions.
    if categories == 0
        || categories >= rows
        || categories > MAX_CATEGORIES
        || rows > i32::MAX as u32
    {
        return None;
    }
    let (counts, valid) = label_counts(m, f.observed, integer, categories, one_based, &r.boundary);
    let size = materialize(m, Expr::Integer(i64::from(categories)));
    let origin = materialize(m, Expr::Integer(i64::from(one_based)));
    let keys = [f.observed, size, origin];
    let counts = r.summary_for(m, &keys, "category_counts", Expr::Node(counts));
    let valid = r.summary_for(m, &keys, "category_support", Expr::Node(valid));
    let p = map(
        "ifelse",
        [
            map("iszero", [counts.clone()]),
            num(1.0),
            Expr::Node(probabilities),
        ],
    );
    let score = dot(counts, map("log", [p]));
    let outside = r.anchor(
        m,
        f,
        &[(name, probabilities)],
        Expr::Integer(i64::from(categories) + 1),
    );
    Some(op("add", [score, op("ifelse", [valid, num(0.0), outside])]))
}
