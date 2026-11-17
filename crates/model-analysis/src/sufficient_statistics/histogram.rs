//! Repeated fixed observations share the original scalar density calculation.

use super::{Expr, Factor, Reducer, call, num, op, resolve};
use flatppl_core::{Module, Node, NodeId, Scalar, ScalarType, Type};

fn fixed_scalar(m: &Module, id: NodeId, reducer: &Reducer) -> Option<Scalar> {
    let id = resolve(m, id, &reducer.boundary);
    match m.node(id) {
        Node::Lit(value @ (Scalar::Int(_) | Scalar::Bool(_))) => Some(value.clone()),
        Node::Lit(Scalar::Real(value)) if !value.is_nan() => Some(Scalar::Real(*value)),
        Node::Const(name) if m.resolve(*name) == "inf" => Some(Scalar::Real(f64::INFINITY)),
        _ => {
            let negative = call(m, id, "neg")?;
            let [value] = *negative.args else { return None };
            match fixed_scalar(m, value, reducer)? {
                Scalar::Int(value) => value.checked_neg().map(Scalar::Int),
                Scalar::Real(value) => Some(Scalar::Real(-value)),
                _ => None,
            }
        }
    }
}

fn identical(a: &Scalar, b: &Scalar) -> bool {
    match (a, b) {
        (Scalar::Real(a), Scalar::Real(b)) => a.to_bits() == b.to_bits(),
        (Scalar::Int(a), Scalar::Int(b)) => a == b,
        (Scalar::Bool(a), Scalar::Bool(b)) => a == b,
        _ => false,
    }
}

fn has_rank(m: &Module, id: NodeId, rank: usize) -> bool {
    matches!(m.type_of(id), Some(Type::Array { shape, elem })
        if shape.len() == rank && matches!(elem.as_ref(), Type::Scalar(_)))
}

pub(super) fn reduce(reducer: &mut Reducer, m: &mut Module, f: &Factor) -> Option<Expr> {
    let observed = resolve(m, f.observed, &reducer.boundary);
    let observations = &call(m, observed, "vector")?.args;
    let Some(Type::Array { elem, .. }) = m.type_of(f.observed).or_else(|| m.type_of(observed))
    else {
        return None;
    };
    let Type::Scalar(element) = elem.as_ref() else {
        return None;
    };
    let element = *element;
    if observations.len() < 2 {
        return None;
    }
    let mut histogram: Vec<(Scalar, NodeId, usize)> = Vec::new();
    for point in observations.iter().copied() {
        let value = fixed_scalar(m, point, reducer)?;
        if let Some((_, _, count)) = histogram.iter_mut().find(|(x, _, _)| identical(x, &value)) {
            *count += 1;
        } else {
            if histogram.len() == 64 {
                return None;
            }
            histogram.push((value, point, 1));
        }
    }
    if histogram.len() * 2 > observations.len() {
        return None;
    }

    let constructor = m.resolve(f.constructor);
    let names = flatppl_infer::distribution_param_names(constructor)?;
    let ranks = flatppl_infer::distribution_param_ranks(constructor)?;
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, value)| {
            let index = names.iter().position(|n| n == m.resolve(name))?;
            let rank = *ranks.get(index)?;
            let value = if rank == 0 {
                reducer.scalar(m, value)?
            } else {
                let resolved = resolve(m, value, &reducer.boundary);
                if has_rank(m, value, rank) || has_rank(m, resolved, rank) {
                    value
                } else {
                    let wrapped = call(m, resolved, "vector")?;
                    let [only] = *wrapped.args else { return None };
                    if !has_rank(m, only, rank) {
                        return None;
                    }
                    only
                }
            };
            Some((name, value))
        })
        .collect::<Option<Vec<_>>>()?;
    if parameters.len() != names.len() {
        return None;
    }
    histogram
        .into_iter()
        .map(|(value, point, count)| {
            // The original broadcast reads each cell after the vector's
            // element promotion, before the primitive's arithmetic.
            let point = match (element, value) {
                (ScalarType::Real, Scalar::Int(_) | Scalar::Bool(_)) => {
                    op("real", [Expr::Node(point)])
                }
                (ScalarType::Integer, Scalar::Bool(_)) => op("integer", [Expr::Node(point)]),
                _ => Expr::Node(point),
            };
            op(
                "mul",
                [num(count as f64), reducer.anchor(m, f, &parameters, point)],
            )
        })
        .reduce(|a, b| op("add", [a, b]))
}
