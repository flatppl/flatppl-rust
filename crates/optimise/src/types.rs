//! Facts about original nodes. Dynamic dimensions and deferred types are not proofs.

use flatppl_core::{Dim, Mass, ScalarType, Type};
use std::collections::HashMap;

use crate::Profile;
use crate::engine::{Fact, FactArg};

pub(super) fn fact(
    name: &'static str,
    term: usize,
    args: impl IntoIterator<Item = FactArg>,
) -> Fact {
    let mut args: Vec<_> = args.into_iter().collect();
    args.insert(0, FactArg::Expr(term));
    Fact { name, args }
}

fn concrete(ty: &Type) -> bool {
    match ty {
        Type::Scalar(_) | Type::RngState => true,
        Type::Array { shape, elem } => {
            shape.iter().all(|d| matches!(d, Dim::Static(_))) && concrete(elem)
        }
        Type::TVector {
            len: Dim::Static(_),
            elem,
        } => concrete(elem),
        Type::Record(fields) => fields.iter().all(|(_, t)| concrete(t)),
        Type::Tuple(fields) => fields.iter().all(concrete),
        Type::Table {
            columns,
            nrows: Dim::Static(_),
        } => columns.iter().all(|(_, t)| concrete(t)),
        Type::Measure { domain, mass } => *mass != Mass::Deferred && concrete(domain),
        _ => false,
    }
}

pub(super) fn dimensions(dims: impl DoubleEndedIterator<Item = i64>) -> FactArg {
    dims.rev()
        .fold(FactArg::Call("MathNil", vec![]), |tail, n| {
            FactArg::Call("MathDim", vec![FactArg::Int(n), tail])
        })
}

fn static_dims(shape: &[Dim]) -> Option<Vec<i64>> {
    (shape.len() <= 16).then_some(())?;
    shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(i64::from(*n)),
            Dim::Dynamic => None,
        })
        .collect()
}

#[derive(Default)]
pub(super) struct TypeFacts {
    // Types include interned field symbols. Compare them in the original module.
    keys: HashMap<Type, i64>,
}

impl TypeFacts {
    pub(super) fn seed(
        &mut self,
        facts: &mut Vec<Fact>,
        ty: &Type,
        term: usize,
        integer: bool,
        profile: Profile,
    ) {
        use FactArg::Int;
        if !concrete(ty) {
            return;
        }
        let next = self.keys.len() as i64;
        let key = *self.keys.entry(ty.clone()).or_insert(next);
        facts.push(fact("type-id", term, [Int(key)]));
        let hlo = matches!(profile, Profile::StableHlo { .. });
        match ty {
            Type::Scalar(ScalarType::Boolean) => facts.push(fact("bool-value", term, [])),
            Type::Scalar(ScalarType::Integer) if integer => facts.push(fact("int-value", term, [])),
            Type::Scalar(ScalarType::Real) => facts.push(fact("real-value", term, [])),
            Type::Measure {
                mass: Mass::Normalized,
                ..
            } => facts.push(fact("portable-normalized", term, [])),
            Type::TVector {
                len: Dim::Static(n),
                elem,
            } if hlo && **elem == Type::Scalar(ScalarType::Real) => {
                facts.push(fact("math-real-row", term, [Int(i64::from(*n))]))
            }
            Type::Array { shape, elem } => {
                let Some(dims) = static_dims(shape) else {
                    return;
                };
                match (&**elem, dims.as_slice()) {
                    (Type::Scalar(ScalarType::Real), &[n]) => {
                        facts.push(fact("real-vector", term, [Int(n)]))
                    }
                    (Type::Scalar(ScalarType::Real), &[r, c]) => {
                        facts.push(fact("real-matrix", term, [Int(r), Int(c)]))
                    }
                    _ => {}
                }
                if hlo {
                    let outer = dimensions(dims.into_iter());
                    match &**elem {
                        Type::Scalar(ScalarType::Real) => {
                            facts.push(fact("math-flat-shape", term, [outer]))
                        }
                        Type::Array { shape, elem } if **elem == Type::Scalar(ScalarType::Real) => {
                            if let Some(inner) = static_dims(shape) {
                                facts.push(fact(
                                    "math-block-shape",
                                    term,
                                    [outer, dimensions(inner.into_iter())],
                                ));
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}
