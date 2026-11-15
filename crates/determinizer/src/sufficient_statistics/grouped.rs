//! Fixed gather partitions reuse scalar-family reductions on occupied groups.

use std::collections::HashMap;

use super::{Expr, Factor, Reducer, call, map, materialize, op, per_cell, resolve};
use flatppl_core::{
    Call, CallHead, Dim, Inputs, Module, Node, NodeId, Ref, Scalar, ScalarType, Type, ValueSet,
};

// A small selector evaluator. It never follows a promoted query input's RHS.
pub(super) fn fixed_indices(m: &Module, id: NodeId, r: &Reducer) -> Option<Vec<i64>> {
    let id = resolve(m, id, &r.boundary);
    if let Node::Lit(Scalar::Int(i)) = m.node(id) {
        return Some(vec![*i]);
    }
    if let Some(c) = call(m, id, "vector") {
        return c
            .args
            .iter()
            .map(|&id| match m.node(resolve(m, id, &r.boundary)) {
                Node::Lit(Scalar::Int(i)) => Some(*i),
                _ => None,
            })
            .collect();
    }
    let (c, origin) = call(m, id, "get")
        .map(|c| (c, 1))
        .or_else(|| call(m, id, "get0").map(|c| (c, 0)))?;
    let [array, selector] = *c.args else {
        return None;
    };
    let array = fixed_indices(m, array, r)?;
    fixed_indices(m, selector, r)?
        .into_iter()
        .map(|i| {
            array
                .get(usize::try_from(i.checked_sub(origin)?).ok()?)
                .copied()
        })
        .collect()
}

fn uses(m: &Module, root: NodeId, local: Ref) -> bool {
    let mut pending = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        match m.node(id) {
            Node::Ref(r) if *r == local => return true,
            Node::Call(c) if matches!(&c.inputs, Some(Inputs::Spec(xs)) if xs.iter().any(|(_,r)| *r==local)) =>
                {}
            node => pending.extend(node.children()),
        }
    }
    false
}

/// Lift only known indexing and reify point expressions with their used inputs.
/// Blind vector substitution would change scalar arithmetic inside a lambda.
pub(super) fn factor(m: &mut Module, id: NodeId, r: &Reducer) -> Option<Factor> {
    if let Some(f) = per_cell(m, id, &r.boundary) {
        return Some(f);
    }
    let sum = call(m, id, "sum")?;
    let [arg] = *sum.args else { return None };
    let broadcast = call(m, resolve(m, arg, &r.boundary), "broadcast")?.clone();
    let (head, operands) = broadcast.args.split_first()?;
    let function = call(m, resolve(m, *head, &r.boundary), "functionof")?.clone();
    let (Some(Inputs::Spec(inputs)), [body]) = (&function.inputs, function.args.as_ref()) else {
        return None;
    };
    if inputs.len() != operands.len() {
        return None;
    }
    let body = call(m, *body, "builtin_logdensityof")?;
    let [kernel, record, point] = *body.args else {
        return None;
    };
    let Node::Const(constructor) = *m.node(kernel) else {
        return None;
    };
    let record = call(m, record, "record")?.clone();
    let operand = |m: &Module, id| {
        inputs
            .iter()
            .position(|(_, r)| matches!(m.node(id),Node::Ref(v) if v==r))
            .map(|i| operands[i])
    };
    let held = |m: &Module, id| {
        if let Some(value) = operand(m, id) {
            let c = call(m, resolve(m, value, &r.boundary), "vector")?;
            let [value] = *c.args else { return None };
            Some(value)
        } else {
            super::data_only(m, id, &r.boundary, &r.boundary).then_some(id)
        }
    };
    let mut parameters = Vec::new();
    for a in &record.named {
        let value = if let Some(value) = operand(m, a.value) {
            value
        } else if let Some((c, head)) = call(m, a.value, "get")
            .map(|c| (c, "get"))
            .or_else(|| call(m, a.value, "get0").map(|c| (c, "get0")))
        {
            let [array, selector] = *c.args else {
                return None;
            };
            let array = held(m, array)?;
            // Resolve a local selector, including fixed g[i] composition.
            let selector = if let Some(value) = operand(m, selector) {
                value
            } else if let Some((c, head)) = call(m, selector, "get")
                .map(|c| (c, "get"))
                .or_else(|| call(m, selector, "get0").map(|c| (c, "get0")))
            {
                let [values, indices] = *c.args else {
                    return None;
                };
                let values = held(m, values)?;
                let indices = operand(m, indices)?;
                materialize(m, op(head, [Expr::Node(values), Expr::Node(indices)]))
            } else {
                selector
            };
            let indices = fixed_indices(m, selector, r)?;
            materialize(
                m,
                op(
                    head,
                    [
                        Expr::Node(array),
                        op("vector", indices.into_iter().map(Expr::Integer)),
                    ],
                ),
            )
        } else if matches!(m.type_of(a.value), Some(Type::Scalar(_)))
            && super::data_only(m, a.value, &r.boundary, &r.boundary)
        {
            a.value
        } else {
            return None;
        };
        crate::kernel::annotate_rebuilt(m, value);
        parameters.push((a.name, value));
    }
    let observed = if let Some(value) = operand(m, point) {
        value
    } else {
        let used = inputs
            .iter()
            .zip(operands)
            .filter(|((_, local), _)| uses(m, point, *local))
            .collect::<Vec<_>>();
        if used.is_empty() {
            return None;
        }
        let inputs = used
            .iter()
            .map(|((name, local), _)| (*name, *local))
            .collect::<Vec<_>>();
        let args = used.iter().map(|(_, value)| **value).collect::<Vec<_>>();
        let head = m.intern("functionof");
        let function = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: vec![point].into(),
            named: vec![].into(),
            inputs: Some(Inputs::Spec(inputs.into())),
        }));
        crate::density::build_call(m, "broadcast", &[vec![function], args].concat())
    };
    crate::kernel::annotate_rebuilt(m, observed);
    Some(Factor {
        constructor,
        parameters,
        observed,
    })
}

enum Parameter {
    Shared(NodeId),
    Gather {
        array: NodeId,
        indices: Vec<i64>,
        elem: Type,
        set: ValueSet,
    },
}

pub(super) fn gather_parts(
    m: &Module,
    value: NodeId,
    r: &Reducer,
) -> Option<(NodeId, NodeId, i64)> {
    let id = resolve(m, value, &r.boundary);
    if let Some((c, origin)) = call(m, id, "get")
        .map(|c| (c, 1))
        .or_else(|| call(m, id, "get0").map(|c| (c, 0)))
    {
        let [array, indices] = *c.args else {
            return None;
        };
        Some((array, indices, origin))
    } else {
        let c = call(m, id, "broadcast")?;
        let [head, array, indices] = *c.args else {
            return None;
        };
        let origin = match m.node(head) {
            Node::Const(s) if m.resolve(*s) == "get" => 1,
            Node::Const(s) if m.resolve(*s) == "get0" => 0,
            _ => return None,
        };
        let wrapped = call(m, resolve(m, array, &r.boundary), "vector")?;
        let [array] = *wrapped.args else { return None };
        Some((array, indices, origin))
    }
}

fn gather(m: &Module, value: NodeId, r: &Reducer, count: usize) -> Option<Parameter> {
    let (array, indices, origin) = gather_parts(m, value, r)?;
    let Type::Array { shape, elem } = m.type_of(array)? else {
        return None;
    };
    let [Dim::Static(size)] = shape.as_ref() else {
        return None;
    };
    if !matches!(
        elem.as_ref(),
        Type::Scalar(ScalarType::Real | ScalarType::Integer)
    ) {
        return None;
    }
    let indices = fixed_indices(m, indices, r)?
        .into_iter()
        .map(|index| index.checked_add(1 - origin))
        .collect::<Option<Vec<_>>>()?;
    if indices.len() != count || indices.iter().any(|&k| k < 1 || k > i64::from(*size)) {
        return None;
    }
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

/// Count each (group, outcome) once at the preparation boundary. The score
/// keeps ordered Bernoulli factors, without a binomial coefficient.
fn runtime_bernoulli(r: &mut Reducer, m: &mut Module, f: &Factor) -> Option<NodeId> {
    if !r.preparing || m.resolve(f.constructor) != "Bernoulli" {
        return None;
    }
    let [(name, probability)] = *f.parameters else {
        return None;
    };
    let (array, indices, origin) = gather_parts(m, probability, r)?;
    let vector_size = |id, kind| match m.type_of(id)? {
        Type::Array { shape, elem } if **elem == Type::Scalar(kind) => match **shape {
            [Dim::Static(size)] => Some(size),
            _ => None,
        },
        _ => None,
    };
    let groups = vector_size(array, ScalarType::Real)?;
    let count = vector_size(indices, ScalarType::Integer)?;
    // Bound scoring code size and keep both counts and half-integer edges
    // exact in the native integer and real types of either precision.
    if !(1..=64).contains(&groups)
        || count < 2
        || groups >= count
        || count > i32::MAX as u32
        || vector_size(f.observed, ScalarType::Boolean)? != count
        || !r.data_only(m, indices)
        || !r.data_only(m, f.observed)
        || fixed_indices(m, indices, r).is_some()
    {
        return None;
    }
    let size = materialize(m, Expr::Integer(i64::from(groups)));
    let base = materialize(m, Expr::Integer(origin));
    // Validate before subtracting the origin, including extreme input labels.
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
    let code = map(
        "add",
        [
            map(
                "mul",
                [map("sub", [checked, Expr::Node(base)]), Expr::Integer(2)],
            ),
            map("integer", [Expr::Node(f.observed)]),
        ],
    );
    let edges = op(
        "vector",
        (0..=2 * groups).map(|i| Expr::Number(f64::from(i) - 0.5)),
    );
    let counts = r.summary_for(
        m,
        &[f.observed, indices, size, base],
        "runtime_bernoulli_counts",
        op("bincounts", [edges, code]),
    );
    let p = op("get", [Expr::Held(0), Expr::Local]);
    let index = op("mul", [Expr::Integer(2), Expr::Local]);
    let failures = op(
        "get",
        [Expr::Held(1), op("sub", [index.clone(), Expr::Integer(1)])],
    );
    let successes = op("get", [Expr::Held(1), index]);
    let no = super::weighted_bernoulli(m, name, p.clone(), failures, false);
    let yes = super::weighted_bernoulli(m, name, p, successes, true);
    let terms = Expr::Cells(
        Box::new(op("add", [no, yes])),
        Box::new(op(
            "indicesof",
            [op("fill", [Expr::Integer(0), Expr::Node(size)])],
        )),
        vec![Expr::Node(array), counts],
    );
    r.count += 1;
    r.rule = Some(("runtime grouped Bernoulli counts", None));
    Some(materialize(m, op("sum", [terms])))
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    let f = factor(m, id, r)?;
    reduce_factor(r, m, &f)
}

pub(super) fn reduce_factor(r: &mut Reducer, m: &mut Module, f: &Factor) -> Option<NodeId> {
    if let Some(value) = runtime_bernoulli(r, m, f) {
        return Some(value);
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
    r.fallback = "parameters need shared scalars or fixed, in-range gather selectors";
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, value)| {
            Some((
                name,
                match r.scalar(m, value) {
                    Some(value) => Parameter::Shared(value),
                    None if m.resolve(name) == "support" => Parameter::Shared(value),
                    None => gather(m, value, r, count)?,
                },
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    if !parameters
        .iter()
        .any(|(_, p)| matches!(p, Parameter::Gather { .. }))
    {
        r.fallback = "unsupported representation, support proof, or numerical range";
        return None;
    }
    if !r.data_only(m, f.observed) {
        r.fallback = "observations depend on a scoring input";
        return None;
    }
    if !r.preparing && !r.fixed(m, f.observed) {
        r.fallback = "runtime observations require explicit preparation for grouping";
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
        r.fallback = "every occupied parameter tuple occurs only once";
        return None;
    }
    let partition = super::GroupPartition {
        observations: count,
        groups: groups.len(),
        singletons: groups
            .iter()
            .filter(|(_, positions)| positions.len() == 1)
            .count(),
    };
    r.fallback = "an occupied group fails the family support or numerical guard";
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
                        let index = key.next()?;
                        if let Some(c) = call(
                            &candidate,
                            resolve(&candidate, *array, &reducer.boundary),
                            "vector",
                        ) {
                            let value = c.args[usize::try_from(index - 1).ok()?];
                            if candidate.type_of(value) == Some(elem) {
                                return Some((*name, value));
                            }
                        }
                        let value = materialize(
                            &mut candidate,
                            op("get", [Expr::Node(*array), Expr::Integer(index)]),
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
            if let Some(value) = super::scalar::reduce_factor(&mut reducer, &mut candidate, &factor)
            {
                reducer.count += 1;
                materialize(&mut candidate, value)
            } else if let Some(value) = reducer.reduce_moment_factor(&mut candidate, &factor) {
                value
            } else if let Some(value) =
                super::histogram::reduce(&mut reducer, &mut candidate, &factor)
            {
                reducer.count += 1;
                materialize(&mut candidate, value)
            } else {
                let Some(value) = super::poisson::reduce(&mut reducer, &mut candidate, &factor)
                else {
                    // Keep the failed guard, but discard every candidate edit.
                    r.fallback = reducer.fallback;
                    return None;
                };
                value
            }
        };
        terms.push(Expr::Node(value));
    }
    let terms = terms
        .into_iter()
        .map(|term| materialize(&mut candidate, term))
        .collect::<Vec<_>>();
    let result = crate::density::fold_add(&mut candidate, &terms);
    *m = candidate;
    *r = reducer;
    r.rule = Some(("grouped scalar summaries", Some(partition)));
    Some(result)
}
