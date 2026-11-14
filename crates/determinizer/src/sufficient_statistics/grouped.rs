//! Fixed gather partitions reuse scalar-family reductions on occupied groups.

use std::collections::HashMap;

use super::{Expr, Factor, Reducer, call, data_only, materialize, op, per_cell, resolve};
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, ScalarType, Type, ValueSet};

enum Parameter {
    Shared(NodeId),
    Gather {
        array: NodeId,
        indices: Vec<i64>,
        elem: Type,
        set: ValueSet,
    },
}

fn gather(m: &Module, value: NodeId, r: &Reducer, count: usize) -> Option<Parameter> {
    let id = resolve(m, value, &r.boundary);
    let (array, indices) = if let Some(c) = call(m, id, "get") {
        let [array, indices] = *c.args else {
            return None;
        };
        (array, indices)
    } else {
        let c = call(m, id, "broadcast")?;
        let [head, array, indices] = *c.args else {
            return None;
        };
        if !matches!(m.node(head),Node::Const(s) if m.resolve(*s)=="get") {
            return None;
        }
        let wrapped = call(m, resolve(m, array, &r.boundary), "vector")?;
        let [array] = *wrapped.args else { return None };
        (array, indices)
    };
    let Type::Array { shape, elem } = m.type_of(array)? else {
        return None;
    };
    let [Dim::Static(size)] = shape.as_ref() else {
        return None;
    };
    if !matches!(elem.as_ref(), Type::Scalar(ScalarType::Real)) {
        return None;
    }
    let c = call(m, resolve(m, indices, &r.boundary), "vector")?;
    if c.args.len() != count {
        return None;
    }
    let indices = c
        .args
        .iter()
        .map(|&id| match m.node(resolve(m, id, &r.boundary)) {
            Node::Lit(Scalar::Int(k)) if *k >= 1 && *k <= i64::from(*size) => Some(*k),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let set = match m.valueset_of(array) {
        Some(ValueSet::CartPow(set, _)) => set.as_ref().clone(),
        _ => ValueSet::Reals,
    };
    Some(Parameter::Gather {
        array,
        indices,
        elem: elem.as_ref().clone(),
        set,
    })
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    let f = per_cell(m, id, &r.boundary)?;
    if !matches!(m.resolve(f.constructor), "Normal" | "Poisson")
        || !data_only(m, f.observed, &r.allowed, &r.boundary)
        || (!r.preparing && !data_only(m, f.observed, &Default::default(), &r.boundary))
    {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(f.observed)?.clone() else {
        return None;
    };
    let [Dim::Static(count)] = shape.as_ref() else {
        return None;
    };
    let count = *count as usize;
    if count < 2 {
        return None;
    }
    let root = resolve(m, f.observed, &r.boundary);
    let literal = call(m, root, "vector").map(|c| c.args.to_vec());
    // Do not rewalk a derived observation DAG once for each occupied group.
    if literal.is_none() && !matches!(m.node(root),Node::Ref(v) if r.boundary.contains(&v.name)) {
        return None;
    }
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, value)| {
            Some((
                name,
                match r.scalar(m, value) {
                    Some(value) => Parameter::Shared(value),
                    None => gather(m, value, r, count)?,
                },
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    if !parameters
        .iter()
        .any(|(_, p)| matches!(p, Parameter::Gather { .. }))
    {
        return None;
    }
    let mut groups = Vec::<(Vec<i64>, Vec<usize>)>::new();
    let mut lookup = HashMap::new();
    for i in 0..count {
        let key = parameters
            .iter()
            .filter_map(|(_, p)| match p {
                Parameter::Gather { indices, .. } => Some(indices[i]),
                _ => None,
            })
            .collect::<Vec<_>>();
        let at = *lookup.entry(key.clone()).or_insert_with(|| {
            groups.push((key, Vec::new()));
            groups.len() - 1
        });
        groups[at].1.push(i);
    }
    if groups.len() == count {
        return None;
    }
    // A later bucket can fail a family-specific numerical guard. Commit the
    // whole partition only after every occupied bucket has a valid reduction.
    let mut candidate = m.clone();
    let mut reducer = r.clone();
    let mut terms = Vec::new();
    for (keys, positions) in groups {
        let mut key = keys.into_iter();
        let values = parameters
            .iter()
            .map(|(name, p)| {
                let value = match p {
                    Parameter::Shared(value) => *value,
                    Parameter::Gather {
                        array, elem, set, ..
                    } => {
                        let value = materialize(
                            &mut candidate,
                            op("get", [Expr::Node(*array), Expr::Integer(key.next()?)]),
                        );
                        candidate.set_type(value, elem.clone());
                        candidate.set_valueset(value, set.clone());
                        value
                    }
                };
                Some((*name, value))
            })
            .collect::<Option<Vec<_>>>()?;
        let observed = if let Some(items) = &literal {
            let items = positions
                .iter()
                .map(|&i| {
                    let value = items[i];
                    if matches!(elem.as_ref(), Type::Scalar(ScalarType::Real))
                        && !matches!(
                            candidate.type_of(value),
                            Some(Type::Scalar(ScalarType::Real))
                        )
                    {
                        op("real", [Expr::Node(value)])
                    } else {
                        Expr::Node(value)
                    }
                })
                .collect::<Vec<_>>();
            materialize(&mut candidate, op("vector", items))
        } else {
            materialize(
                &mut candidate,
                op(
                    "get",
                    [
                        Expr::Node(f.observed),
                        op(
                            "vector",
                            positions.iter().map(|i| Expr::Integer(*i as i64 + 1)),
                        ),
                    ],
                ),
            )
        };
        let size = Dim::Static(u32::try_from(positions.len()).ok()?);
        candidate.set_type(
            observed,
            Type::Array {
                shape: vec![size].into(),
                elem: elem.clone(),
            },
        );
        if let Some(ValueSet::CartPow(set, _)) = m.valueset_of(f.observed) {
            candidate.set_valueset(observed, ValueSet::CartPow(set.clone(), size));
        }
        let factor = Factor {
            constructor: f.constructor,
            parameters: values,
            observed,
        };
        let value = if positions.len() == 1 {
            let point = reducer.summary(
                &mut candidate,
                observed,
                "grouped_singleton",
                op("get", [Expr::Node(observed), Expr::Integer(1)]),
            );
            let value = reducer.anchor(&mut candidate, &factor, &factor.parameters, point);
            reducer.count += 1;
            materialize(&mut candidate, value)
        } else {
            reducer
                .reduce_moment_factor(&mut candidate, &factor)
                .or_else(|| super::poisson::reduce(&mut reducer, &mut candidate, &factor))?
        };
        terms.push(Expr::Node(value));
    }
    let result = materialize(&mut candidate, op("sum", [op("vector", terms)]));
    *m = candidate;
    *r = reducer;
    Some(result)
}
