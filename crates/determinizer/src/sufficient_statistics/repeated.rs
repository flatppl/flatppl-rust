//! Reuse primitive predictor evaluations for repeated fixed pointwise rows.

use std::collections::HashMap;

use super::{Expr, Factor, Reducer, call, map, materialize, op, per_cell, resolve};
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, ScalarType, Type, ValueSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Atom {
    Integer(i64),
    Real(u64),
    Boolean(bool),
}

enum Plan {
    Scalar(NodeId),
    Column(usize),
    Apply(&'static str, Vec<usize>),
    Gather(&'static str, NodeId, usize),
}

struct Column {
    values: Vec<NodeId>,
    keys: Vec<Atom>,
    real: bool,
}

fn atom(m: &Module, r: &Reducer, id: NodeId, depth: usize) -> Option<Atom> {
    if depth > 64 {
        return None;
    }
    let id = resolve(m, id, &r.boundary);
    match m.node(id) {
        Node::Lit(Scalar::Int(value)) => Some(Atom::Integer(*value)),
        Node::Lit(Scalar::Real(value)) if value.is_finite() => Some(Atom::Real(value.to_bits())),
        Node::Lit(Scalar::Bool(value)) => Some(Atom::Boolean(*value)),
        _ => {
            let c = call(m, id, "neg")?;
            let [value] = *c.args else { return None };
            match atom(m, r, value, depth + 1)? {
                Atom::Integer(value) => value.checked_neg().map(Atom::Integer),
                Atom::Real(value) => Some(Atom::Real((-f64::from_bits(value)).to_bits())),
                Atom::Boolean(_) => None,
            }
        }
    }
}

#[derive(Default)]
struct Predictor {
    plans: Vec<Plan>,
    columns: Vec<Column>,
    visited: HashMap<NodeId, usize>,
}

fn pointwise(name: &str) -> Option<&'static str> {
    Some(match name {
        "add" => "add",
        "sub" => "sub",
        "mul" => "mul",
        "divide" => "divide",
        "neg" => "neg",
        "exp" => "exp",
        "invlogit" => "invlogit",
        "log" => "log",
        "log1p" => "log1p",
        "sqrt" => "sqrt",
        "abs" => "abs",
        "pow" => "pow",
        _ => return None,
    })
}

impl Predictor {
    fn visit(
        &mut self,
        m: &Module,
        r: &Reducer,
        id: NodeId,
        count: u32,
        depth: usize,
    ) -> Option<usize> {
        let id = resolve(m, id, &r.boundary);
        if let Some(&plan) = self.visited.get(&id) {
            return Some(plan);
        }
        if depth > 64 {
            return None;
        }
        let plan = if let Some(scalar) = r
            .scalar(m, id)
            .filter(|&id| matches!(m.type_of(id), Some(Type::Scalar(_))))
        {
            Plan::Scalar(scalar)
        } else if let Some(c) = call(m, id, "vector") {
            let Type::Array { shape, elem } = m.type_of(id)? else {
                return None;
            };
            if shape.as_ref() != [Dim::Static(count)] || c.args.len() != count as usize {
                return None;
            }
            let keys = c
                .args
                .iter()
                .map(|&value| atom(m, r, value, 0))
                .collect::<Option<Vec<_>>>()?;
            let column = self.columns.len();
            self.columns.push(Column {
                values: c.args.to_vec(),
                keys,
                real: **elem == Type::Scalar(ScalarType::Real),
            });
            Plan::Column(column)
        } else if let Some((array, indices, origin)) = super::grouped::gather_parts(m, id, r) {
            let Type::Array { shape, elem } = m.type_of(array)? else {
                return None;
            };
            let [Dim::Static(size)] = shape.as_ref() else {
                return None;
            };
            if !matches!(elem.as_ref(), Type::Scalar(_)) {
                return None;
            }
            let fixed = super::grouped::fixed_indices(m, indices, r)?;
            if fixed
                .iter()
                .any(|&index| index < origin || index - origin >= i64::from(*size))
            {
                return None;
            }
            let selector = self.visit(m, r, indices, count, depth + 1)?;
            Plan::Gather(if origin == 0 { "get0" } else { "get" }, array, selector)
        } else {
            let c = call(m, id, "broadcast")?;
            let (head, args) = c.args.split_first()?;
            let Node::Const(head) = m.node(*head) else {
                return None;
            };
            let operation = pointwise(m.resolve(*head))?;
            let operands = args
                .iter()
                .map(|&value| self.visit(m, r, value, count, depth + 1))
                .collect::<Option<Vec<_>>>()?;
            Plan::Apply(operation, operands)
        };
        let at = self.plans.len();
        self.plans.push(plan);
        self.visited.insert(id, at);
        Some(at)
    }

    fn rebuild(
        &self,
        m: &mut Module,
        at: usize,
        rows: &[usize],
        cache: &mut [Option<NodeId>],
    ) -> NodeId {
        if let Some(node) = cache[at] {
            return node;
        }
        let value = match &self.plans[at] {
            Plan::Scalar(id) => *id,
            Plan::Column(column) => {
                let column = &self.columns[*column];
                let values = rows.iter().map(|&i| match column.keys[i] {
                    Atom::Integer(value) if column.real => Expr::Number(value as f64),
                    _ => Expr::Node(column.values[i]),
                });
                materialize(m, op("vector", values))
            }
            Plan::Apply(name, args) => {
                let args = args
                    .iter()
                    .map(|&i| Expr::Node(self.rebuild(m, i, rows, cache)))
                    .collect::<Vec<_>>();
                materialize(m, map(name, args))
            }
            Plan::Gather(name, array, indices) => {
                let indices = self.rebuild(m, *indices, rows, cache);
                materialize(m, op(name, [Expr::Node(*array), Expr::Node(indices)]))
            }
        };
        crate::kernel::annotate_rebuilt(m, value);
        cache[at] = Some(value);
        value
    }
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    let f = per_cell(m, id, &r.boundary)?;
    if !r.data_only(m, f.observed) || (!r.preparing && !r.fixed(m, f.observed)) {
        return None;
    }
    // Fixed predictors already fold in backends. Only compress live scoring
    // work, rather than introducing summaries for constant constructor arrays.
    if f.parameters.iter().all(|&(_, value)| r.data_only(m, value)) {
        return None;
    }
    let Type::Array { shape, .. } = m.type_of(f.observed)? else {
        return None;
    };
    let [Dim::Static(count)] = **shape else {
        return None;
    };
    if count < 2 {
        return None;
    }
    let mut predictor = Predictor::default();
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, value)| Some((name, value, predictor.visit(m, r, value, count, 0)?)))
        .collect::<Option<Vec<_>>>()?;
    if predictor.columns.is_empty() {
        return None;
    }
    let mut lookup = HashMap::new();
    let mut rows = Vec::new();
    let mut indices = Vec::new();
    for i in 0..count as usize {
        let key = predictor
            .columns
            .iter()
            .map(|c| c.keys[i])
            .collect::<Vec<_>>();
        let next = rows.len();
        let at = *lookup.entry(key).or_insert_with(|| {
            rows.push(i);
            next
        });
        if rows.len() > 64 {
            return None;
        }
        indices.push(at as i64 + 1);
    }
    if rows.len() == count as usize {
        return None;
    }
    let mut candidate = m.clone();
    let mut reducer = r.clone();
    let mut cache = vec![None; predictor.plans.len()];
    let selectors = materialize(
        &mut candidate,
        op("vector", indices.into_iter().map(Expr::Integer)),
    );
    let parameters = parameters
        .into_iter()
        .map(|(name, old, plan)| {
            let values = predictor.rebuild(&mut candidate, plan, &rows, &mut cache);
            if matches!(candidate.type_of(values), Some(Type::Scalar(_))) {
                return (name, values);
            }
            let old_type = m.type_of(old).unwrap().clone();
            if let Some(ValueSet::CartPow(set, _)) = m.valueset_of(old) {
                candidate.set_valueset(
                    values,
                    ValueSet::CartPow(set.clone(), Dim::Static(rows.len() as u32)),
                );
            }
            let value = materialize(
                &mut candidate,
                op("get", [Expr::Node(values), Expr::Node(selectors)]),
            );
            candidate.set_type(value, old_type);
            if let Some(set) = m.valueset_of(old) {
                candidate.set_valueset(value, set.clone());
            }
            (name, value)
        })
        .collect();
    let factor = Factor {
        constructor: f.constructor,
        parameters,
        observed: f.observed,
    };
    let result = super::grouped::reduce_factor(&mut reducer, &mut candidate, &factor)?;
    let partition = reducer
        .rule
        .as_ref()
        .and_then(|(_, partition)| partition.clone());
    reducer.rule = Some(("repeated predictor rows", partition));
    *m = candidate;
    *r = reducer;
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_atoms_preserve_real_bits_and_scalar_types() {
        let keys = [
            Atom::Real(0.0_f64.to_bits()),
            Atom::Real((-0.0_f64).to_bits()),
            Atom::Integer(0),
            Atom::Boolean(false),
        ];
        assert_eq!(
            keys.into_iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            4
        );
    }
}
