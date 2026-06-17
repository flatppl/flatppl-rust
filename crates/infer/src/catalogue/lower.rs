//! The single code↔data bridge: a `Sig` + call context → core inference types.
//! Arg-dependent forms (VectorFromParam, DimExpr) are evaluated here.

use super::{DimExpr, DomainSig, MassTag, ResultSig, ScalarTag, Sig, SupportTag};
use flatppl_core::{Dim, Mass, ScalarType, Type, ValueSet};

/// What `lower` needs from the call site to evaluate arg-dependent signatures.
pub(crate) struct LowerCtx<'a> {
    pub(crate) arg_scalar: &'a dyn Fn(usize) -> Option<ScalarType>,
    pub(crate) param_dim: &'a dyn Fn(&str) -> Dim,
    pub(crate) arg_dim: &'a dyn Fn(usize) -> Dim,
}

fn scalar(tag: ScalarTag) -> ScalarType {
    match tag {
        ScalarTag::Real => ScalarType::Real,
        ScalarTag::Integer => ScalarType::Integer,
        ScalarTag::Boolean => ScalarType::Boolean,
        ScalarTag::Complex => ScalarType::Complex,
    }
}

fn support(tag: SupportTag) -> ValueSet {
    match tag {
        SupportTag::Reals => ValueSet::Reals,
        SupportTag::PosReals => ValueSet::PosReals,
        SupportTag::NonNegReals => ValueSet::NonNegReals,
        SupportTag::UnitInterval => ValueSet::UnitInterval,
        SupportTag::Integers => ValueSet::Integers,
        SupportTag::PosIntegers => ValueSet::PosIntegers,
        SupportTag::NonNegIntegers => ValueSet::NonNegIntegers,
        SupportTag::Booleans => ValueSet::Booleans,
        SupportTag::Complexes => ValueSet::Complexes,
        SupportTag::Anything => ValueSet::Anything,
        // Dim-aware tags: caller must use `support_ctx` for these.
        SupportTag::StdSimplex
        | SupportTag::CartPowReals
        | SupportTag::CartPowNonNegIntegers
        | SupportTag::Unknown
        // Structural: the live support is computed from call args (ops.rs
        // `distribution_support`); Unknown is the correct static approximation.
        | SupportTag::Structural => ValueSet::Unknown,
    }
}

/// Resolve a support tag that may depend on `param` (for VectorFromParam sigs).
fn support_ctx(tag: SupportTag, param: &str, ctx: &LowerCtx) -> ValueSet {
    match tag {
        SupportTag::StdSimplex => ValueSet::StdSimplex((ctx.param_dim)(param)),
        SupportTag::CartPowReals => {
            ValueSet::CartPow(Box::new(ValueSet::Reals), (ctx.param_dim)(param))
        }
        SupportTag::CartPowNonNegIntegers => {
            let d = (ctx.param_dim)(param);
            ValueSet::CartPow(Box::new(ValueSet::NonNegIntegers), d)
        }
        SupportTag::Unknown => ValueSet::Unknown,
        other => support(other),
    }
}

/// Like `support_ctx` for DynMatrix distributions (no named param, dim always dynamic).
fn support_dyn(tag: SupportTag) -> ValueSet {
    match tag {
        SupportTag::Unknown => ValueSet::Unknown,
        other => support(other),
    }
}

fn mass(tag: MassTag) -> Mass {
    match tag {
        MassTag::Normalized => Mass::Normalized,
        MassTag::Finite => Mass::Finite,
        MassTag::LocallyFinite => Mass::LocallyFinite,
        MassTag::Unknown => Mass::Unknown,
    }
}

/// Lower a signature to `(Type, ValueSet)`. Phase is supplied by the caller.
pub(crate) fn lower(sig: &Sig, ctx: &LowerCtx) -> (Type, ValueSet) {
    match sig {
        Sig::Distribution {
            domain,
            support: sup,
            mass: m,
        } => {
            let (dom, vset) = match domain {
                DomainSig::Scalar(s) => (Type::Scalar(scalar(*s)), support(*sup)),
                DomainSig::VectorFromParam { elem, param } => {
                    let ty = Type::Array {
                        shape: Box::new([(ctx.param_dim)(param)]),
                        elem: Box::new(Type::Scalar(scalar(*elem))),
                    };
                    let vs = support_ctx(*sup, param, ctx);
                    (ty, vs)
                }
                DomainSig::DynMatrix => {
                    let ty = Type::Array {
                        shape: Box::new([Dim::Dynamic, Dim::Dynamic]),
                        elem: Box::new(Type::Scalar(ScalarType::Real)),
                    };
                    (ty, support_dyn(*sup))
                }
            };
            (
                Type::Measure {
                    domain: Box::new(dom),
                    mass: mass(*m),
                },
                vset,
            )
        }
        Sig::Function { params: _, result } => {
            let ty = match result {
                ResultSig::Scalar(s) => Type::Scalar(scalar(*s)),
                ResultSig::SameScalarKind(i) => {
                    Type::Scalar((ctx.arg_scalar)(*i).unwrap_or(ScalarType::Real))
                }
                ResultSig::DomainMap { arg, map } => {
                    let got = (ctx.arg_scalar)(*arg);
                    let out = map
                        .iter()
                        .find_map(|(inp, outp)| {
                            (Some(scalar(*inp)) == got).then_some(scalar(*outp))
                        })
                        .unwrap_or(ScalarType::Real);
                    Type::Scalar(out)
                }
                ResultSig::Matrix { rows, cols } => {
                    let d = |e: &DimExpr| match e {
                        DimExpr::Dyn => Dim::Dynamic,
                        DimExpr::OfParam(i) => (ctx.arg_dim)(*i),
                        DimExpr::MulDims(_, _) => Dim::Dynamic, // shape arithmetic degraded → dynamic
                    };
                    Type::Array {
                        shape: Box::new([d(rows), d(cols)]),
                        elem: Box::new(Type::Scalar(ScalarType::Real)),
                    }
                }
            };
            let vset = ValueSet::natural_of(&ty);
            (ty, vset)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_a_distribution() {
        let sig = Sig::Distribution {
            domain: DomainSig::Scalar(ScalarTag::Real),
            support: SupportTag::Reals,
            mass: MassTag::Normalized,
        };
        let cx = LowerCtx {
            arg_scalar: &|_| Some(ScalarType::Real),
            param_dim: &|_| Dim::Dynamic,
            arg_dim: &|_| Dim::Dynamic,
        };
        let (ty, vset) = lower(&sig, &cx);
        assert!(matches!(
            ty,
            Type::Measure {
                mass: Mass::Normalized,
                ..
            }
        ));
        assert_eq!(vset, ValueSet::Reals);
    }

    #[test]
    fn same_scalar_kind_follows_arg() {
        let sig = Sig::Function {
            params: vec![],
            result: ResultSig::SameScalarKind(0),
        };
        let cx = LowerCtx {
            arg_scalar: &|_| Some(ScalarType::Complex),
            param_dim: &|_| Dim::Dynamic,
            arg_dim: &|_| Dim::Dynamic,
        };
        let (ty, _) = lower(&sig, &cx);
        assert_eq!(ty, Type::Scalar(ScalarType::Complex));
    }
}
