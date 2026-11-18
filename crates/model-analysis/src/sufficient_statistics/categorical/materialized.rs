//! Revisit a generated label histogram once file observations become literals.

use super::{
    MAX_CATEGORIES, Module, Node, NodeId, Scalar, Symbol, call, fixed_counts, fixed_values, resolve,
};
use crate::sufficient_statistics::{Expr, map, materialize, multivariate::vector};
use flatppl_core::{ScalarType, Type};
use std::collections::HashSet;

struct Graph<'a> {
    m: &'a Module,
    boundary: &'a HashSet<Symbol>,
}

impl<'a> Graph<'a> {
    fn root(&self, id: NodeId) -> NodeId {
        resolve(self.m, id, self.boundary)
    }

    fn args<const N: usize>(&self, id: NodeId, name: &str) -> Option<[NodeId; N]> {
        call(self.m, self.root(id), name)?
            .args
            .as_ref()
            .try_into()
            .ok()
    }

    fn mapped<const N: usize>(&self, id: NodeId, name: &str) -> Option<[NodeId; N]> {
        let call = call(self.m, self.root(id), "broadcast")?;
        let (head, args) = call.args.split_first()?;
        matches!(self.m.node(*head), Node::Const(symbol) if self.m.resolve(*symbol) == name)
            .then(|| args.try_into().ok())?
    }

    fn number(&self, id: NodeId) -> Option<f64> {
        match self.m.node(self.root(id)) {
            Node::Lit(Scalar::Int(n)) => Some(*n as f64),
            Node::Lit(Scalar::Real(x)) => Some(*x),
            _ => {
                let [inner] = self.args(id, "neg")?;
                match self.m.node(self.root(inner)) {
                    Node::Lit(Scalar::Int(n)) => Some(-(*n as f64)),
                    _ => None,
                }
            }
        }
    }

    fn same(&self, a: NodeId, b: NodeId) -> bool {
        self.m.structural_eq(self.root(a), self.root(b))
    }

    /// Read the complete range and integrality check constructed by label_counts.
    fn range(&self, flags: NodeId) -> Option<(NodeId, i64, u32, bool)> {
        let [low, high] = self.mapped(flags, "land")?;
        let (low, high, whole) = match self.mapped(low, "land") {
            Some([a, b]) => (a, b, Some(high)),
            None => (low, high, None),
        };
        let [data, first] = self.mapped(low, "ge")?;
        let [also_data, last] = self.mapped(high, "le")?;
        if !self.same(data, also_data) {
            return None;
        }
        if let Some(whole) = whole {
            let [lhs, floor] = self.mapped(whole, "le")?;
            let [rhs] = self.mapped(floor, "floor")?;
            if !self.same(data, lhs) || !self.same(data, rhs) {
                return None;
            }
        }
        let first = self.number(first)?;
        let d = self.number(last)? - first + 1.;
        if !matches!(first, 0. | 1.)
            || !(1.0..=f64::from(MAX_CATEGORIES)).contains(&d)
            || d.fract() != 0.
        {
            return None;
        }
        let integer = matches!(self.m.type_of(data), Some(Type::Array { elem, .. })
            if matches!(elem.as_ref(), Type::Scalar(ScalarType::Integer)));
        (integer || whole.is_some()).then_some((data, first as i64, d as u32, integer))
    }
}

/// The materializer may introduce aliases for shared broadcasts. Respect every
/// input boundary and fold only the complete support or histogram expression.
pub(in crate::sufficient_statistics) fn fold(
    m: &mut Module,
    rhs: NodeId,
    boundary: &HashSet<Symbol>,
) -> Option<NodeId> {
    let graph = Graph { m, boundary };
    let cast = graph.mapped::<1>(rhs, "real");
    let root = cast.map_or(rhs, |[inner]| inner);
    let (data, first, d, integer, histogram) = if let Some([flags]) = graph.args(root, "lall") {
        if cast.is_some() {
            return None;
        }
        let (data, first, d, integer) = graph.range(flags)?;
        (data, first, d, integer, false)
    } else {
        let [edges, labels] = graph.args(root, "bincounts")?;
        let [start, stop, size] = graph.args(edges, "linspace")?;
        let shifted = graph.mapped::<2>(labels, "sub");
        let selected = shifted.map_or(labels, |[selected, _]| selected);
        let [flags, raw, below] = graph.mapped(selected, "ifelse")?;
        let (data, first, d, integer) = graph.range(flags)?;
        let shift = match shifted {
            Some([_, shift]) => graph.number(shift)?,
            None => 0.,
        };
        if !graph.same(data, raw)
            || shift != first as f64
            || graph.number(below)? != (first - 1) as f64
            || graph.number(start)? != 0.
            || graph.number(stop)? != f64::from(d)
            || graph.number(size)? != f64::from(d) + 1.
        {
            return None;
        }
        (data, first, d, integer, true)
    };
    let values = fixed_values(m, data, boundary)?;
    let (counts, valid) = fixed_counts(values, d, first, integer)?;
    if !histogram {
        return Some(m.alloc(Node::Lit(Scalar::Bool(valid))));
    }
    let counts = vector(counts.into_iter().map(Expr::Integer));
    Some(materialize(
        m,
        if cast.is_some() {
            map("real", [counts])
        } else {
            counts
        },
    ))
}
