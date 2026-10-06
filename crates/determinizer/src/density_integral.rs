//! Positive one-dimensional integrals in the compiler's internal IR.
//! Exact rules run first. Ordinary FlatPDL conformance rejects this extension.

use super::*;

fn point(m: &mut Module) -> NodeId {
    let token = m.alloc(Node::Lit(Scalar::Int(m.node_count() as i64)));
    let point = build_call(m, flatppl_core::INTEGRATION_POINT, &[token]);
    m.set_type(point, Type::Scalar(ScalarType::Real));
    m.set_phase(point, flatppl_core::Phase::Fixed);
    point
}

/// Integrate one named, continuous latent shared by a conditional measure.
pub(super) fn marginal(
    m: &mut Module,
    measure: NodeId,
    value: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    let ancestors = measure_stochastic_ancestors(m, measure, &[]);
    let Ancestor::Named { name, prior } = *ancestors.first()? else {
        return None;
    };
    if ancestors
        .iter()
        .any(|a| !matches!(a, Ancestor::Named { name: other, .. } if *other == name))
        || measure_reaches_draw(m, prior, &[])
    {
        return None;
    }
    let (lo, hi) = bounds(m, prior)?;
    let coordinate = point(m);
    m.push_conditioned_inputs(&[name]);
    let conditional = lower_measure_density(m, measure, value);
    m.pop_conditioned_inputs(1);
    Some(conditional.and_then(|conditional| {
        let conditional = crate::kernel::substitute_admitted(
            m,
            conditional,
            &[(
                Ref {
                    ns: RefNs::SelfMod,
                    name,
                },
                coordinate,
            )],
            crate::kernel::Substitute::All,
            true,
        );
        let prior = lower_measure_density(m, prior, coordinate)?;
        let integrand = build_call(m, "add", &[conditional, prior]);
        Ok(build_call(
            m,
            flatppl_core::LOG_INTEGRAL,
            &[integrand, coordinate, lo, hi],
        ))
    }))
}

pub(super) fn mass(m: &mut Module, measure: NodeId) -> Option<Result<NodeId, RefuseError>> {
    if measure_reaches_draw(m, measure, &[]) {
        return None;
    }
    let (lo, hi) = bounds(m, measure)?;
    let coordinate = point(m);
    Some(
        lower_measure_density(m, measure, coordinate).map(|integrand| {
            build_call(
                m,
                flatppl_core::LOG_INTEGRAL,
                &[integrand, coordinate, lo, hi],
            )
        }),
    )
}

/// A conservative scalar support interval. Preserve dynamic interval endpoints.
fn bounds(m: &mut Module, measure: NodeId) -> Option<(NodeId, NodeId)> {
    let (node, _) = resolve_ref_chain(m, measure);
    if !continuous(m, node)
        || !matches!(m.type_of(node), Some(Type::Measure { domain, .. })
        if matches!(domain.as_ref(), Type::Scalar(ScalarType::Real)))
    {
        return None;
    }
    if let Node::Call(c) = m.node(node).clone()
        && let CallHead::Builtin(head) = c.head
    {
        let support = match m.resolve(head) {
            "truncate" => c.args.get(1).copied(),
            "Lebesgue" | "Uniform" => named_or_positional(m, &c, "support"),
            "weighted" | "logweighted" if c.args.len() == 2 => return bounds(m, c.args[1]),
            _ => None,
        };
        if let Some(set) = support {
            let (set, _) = resolve_ref_chain(m, set);
            if let Some(interval) = expect_builtin_call(m, set, "interval")
                && interval.args.len() == 2
            {
                return Some((interval.args[0], interval.args[1]));
            }
        }
    }
    let (lo, hi) = match m.valueset_of(node)? {
        ValueSet::Reals => (f64::NEG_INFINITY, f64::INFINITY),
        ValueSet::PosReals | ValueSet::NonNegReals => (0.0, f64::INFINITY),
        ValueSet::UnitInterval => (0.0, 1.0),
        ValueSet::Interval(lo, hi) => (*lo, *hi),
        _ => return None,
    };
    Some((
        m.alloc(Node::Lit(Scalar::Real(lo))),
        m.alloc(Node::Lit(Scalar::Real(hi))),
    ))
}

/// A real support does not prove absolute continuity: Dirac and mixtures can
/// have atoms there. Admit only constructors with a scalar Lebesgue density.
fn continuous(m: &Module, measure: NodeId) -> bool {
    let (node, _) = resolve_ref_chain(m, measure);
    let Node::Call(c) = m.node(node) else {
        return false;
    };
    let CallHead::Builtin(head) = c.head else {
        return false;
    };
    match m.resolve(head) {
        "Lebesgue" | "Uniform" | "Normal" | "Exponential" | "Gamma" | "InverseGamma" | "Beta"
        | "StudentT" | "Cauchy" | "Laplace" | "Logistic" | "LogNormal" | "Weibull"
        | "ChiSquared" | "GeneralizedNormal" => true,
        "weighted" | "logweighted" if c.args.len() == 2 => continuous(m, c.args[1]),
        "truncate" | "normalize" => c.args.first().is_some_and(|&base| continuous(m, base)),
        _ => false,
    }
}
