//! Recognize normalized restrictions and hurdle laws for native sampling.

use super::*;
use flatppl_core::RestrictedSample;

pub(super) fn parameters(
    m: &mut Module,
    measure: NodeId,
) -> Option<(Symbol, Vec<(Symbol, NodeId)>)> {
    let inner = normalized_inner(m, measure)?;
    let (sampler, base, extra) = if let Some((sampler, base, lo, hi)) = restriction(m, inner) {
        let extra = match sampler {
            RestrictedSample::NormalInterval => vec![("lower", lo), ("upper", hi)],
            _ => Vec::new(),
        };
        (sampler, base, extra)
    } else {
        let (sampler, base, zero, positive) = hurdle(m, inner)?;
        (
            sampler,
            base,
            vec![("zero_weight", zero), ("positive_weight", positive)],
        )
    };
    let (_, mut params) = split_kernel_constructor(m, base)?;
    params.extend(
        extra
            .into_iter()
            .map(|(name, value)| (m.intern(name), value)),
    );
    Some((m.intern(sampler.tag()), params))
}

fn normalized_inner(m: &Module, measure: NodeId) -> Option<NodeId> {
    let [inner] = *expect_builtin_call(m, resolve_ref(m, measure).0, "normalize")?.args else {
        return None;
    };
    Some(resolve_ref(m, inner).0)
}

fn restriction(m: &Module, measure: NodeId) -> Option<(RestrictedSample, NodeId, NodeId, NodeId)> {
    let [base, support] = *expect_builtin_call(m, measure, "truncate")?.args else {
        return None;
    };
    let [lo, hi] = *expect_builtin_call(m, resolve_ref(m, support).0, "interval")?.args else {
        return None;
    };
    let base = resolve_ref(m, base).0;
    let positive = || {
        crate::density::static_real(m, lo) == Some(1.0)
            && crate::density::static_real(m, hi) == Some(f64::INFINITY)
    };
    let sampler = match builtin_name(m, base)? {
        "Normal" => RestrictedSample::NormalInterval,
        "Poisson" if positive() => RestrictedSample::PositivePoisson,
        "NegativeBinomial" if positive() => RestrictedSample::PositiveNegativeBinomial,
        _ => return None,
    };
    Some((sampler, base, lo, hi))
}

fn hurdle(m: &Module, measure: NodeId) -> Option<(RestrictedSample, NodeId, NodeId, NodeId)> {
    let [first, second] = *expect_builtin_call(m, measure, "superpose")?.args else {
        return None;
    };
    let weighted = |node| {
        let [weight, base] = *expect_builtin_call(m, resolve_ref(m, node).0, "weighted")?.args
        else {
            return None;
        };
        if crate::density::weight_is_variate_dependent(m, weight) {
            return None;
        }
        Some((weight, resolve_ref(m, base).0))
    };
    let is_zero = |node| {
        let Some(call) = expect_builtin_call(m, node, "Dirac") else {
            return false;
        };
        let [atom] = *call.args else {
            return false;
        };
        matches!(m.node(resolve_ref(m, atom).0), Node::Lit(Scalar::Int(0)))
    };
    let (first_weight, first_base) = weighted(first)?;
    let (second_weight, second_base) = weighted(second)?;
    let (zero_weight, positive_weight, positive_base) = if is_zero(first_base) {
        (first_weight, second_weight, second_base)
    } else if is_zero(second_base) {
        (second_weight, first_weight, first_base)
    } else {
        return None;
    };
    let (sampler, base, ..) = restriction(m, normalized_inner(m, positive_base)?)?;
    let hurdle = match sampler {
        RestrictedSample::PositivePoisson => RestrictedSample::HurdlePoisson,
        RestrictedSample::PositiveNegativeBinomial => RestrictedSample::HurdleNegativeBinomial,
        _ => return None,
    };
    Some((hurdle, base, zero_weight, positive_weight))
}
