//! Evidence for closed conjugate predictive density queries.

use std::collections::HashSet;

use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{
    Call, Inputs, Mass, Module, NamedKind, Node, NodeId, Phase, RefNs, Scalar, ScalarType, Type,
};

use super::engine::{Fact, FactArg};

fn positional<'a>(m: &'a Module, id: NodeId, head: &str) -> Option<&'a [NodeId]> {
    let call = expect_builtin_call(m, id, head)?;
    (call.named.is_empty() && call.inputs.is_none() && !m.is_sample_result(id))
        .then_some(call.args.as_ref())
}

fn parameters<const N: usize>(m: &Module, call: &Call, names: [&str; N]) -> Option<[NodeId; N]> {
    if call.inputs.is_some() || call.args.len() + call.named.len() != N {
        return None;
    }
    let mut values = [None; N];
    for (slot, &value) in values.iter_mut().zip(&call.args) {
        *slot = Some(value);
    }
    for named in &call.named {
        let position = names
            .iter()
            .position(|&name| name == m.resolve(named.name))?;
        if named.kind != NamedKind::Kwarg || values[position].replace(named.value).is_some() {
            return None;
        }
    }
    Some(values.map(|value| value.expect("each parameter must bind exactly once")))
}

fn normalized_scalar(m: &Module, id: NodeId, kind: ScalarType) -> bool {
    matches!(m.type_of(id), Some(Type::Measure { domain, mass: Mass::Normalized })
        if **domain == Type::Scalar(kind))
}

fn fixed_real(m: &Module, id: NodeId, resolve: &impl Fn(NodeId) -> Option<NodeId>) -> Option<f64> {
    let resolved = resolve(id)?;
    let (id, sign) = match positional(m, resolved, "neg") {
        Some(&[literal]) => (resolve(literal)?, -1.0),
        _ => (resolved, 1.0),
    };
    if [resolved, id].into_iter().any(|id| {
        m.is_sample_result(id)
            || m.sample_origin(id).is_some()
            || m.phase_of(id) != Some(Phase::Fixed)
    }) {
        return None;
    }
    let value = match m.node(id) {
        Node::Lit(Scalar::Real(value)) => *value,
        Node::Lit(Scalar::Int(value)) => {
            // The 32-bit integer emitter must accept the original literal.
            f64::from(i32::try_from(*value).ok()?)
        }
        _ => return None,
    };
    (value.is_finite() && f64::from(value as f32) == value).then_some(sign * value)
}

fn positive_normal(value: f64) -> Option<f64> {
    (value >= f64::from(f32::MIN_POSITIVE)).then_some(value)
}

fn binary_point(
    m: &Module,
    point: NodeId,
    resolve: &impl Fn(NodeId) -> Option<NodeId>,
) -> Option<()> {
    // The generic beta-binomial log-gamma expression can have poles beyond
    // its support. Prove binary outcomes without trusting a runtime input's
    // nominal value set or erasing its authored condition.
    let mut pending = vec![point];
    let mut seen = HashSet::new();
    while let Some(id) = pending.pop() {
        let id = resolve(id)?;
        if !seen.insert(id) {
            continue;
        }
        if seen.len() > 32 || m.is_sample_result(id) || m.sample_origin(id).is_some() {
            return None;
        }
        if matches!(m.node(id), Node::Lit(Scalar::Int(0 | 1))) {
            continue;
        }
        let &[condition, yes, no] = positional(m, id, "ifelse")? else {
            return None;
        };
        if m.type_of(condition) != Some(&Type::Scalar(ScalarType::Boolean))
            || m.type_of(id) != Some(&Type::Scalar(ScalarType::Integer))
        {
            return None;
        }
        pending.extend([yes, no]);
    }
    Some(())
}

fn well_conditioned_binomial(a: f64, b: f64, p: f64) -> bool {
    // The fallback subtracts two log-beta values, each built from three
    // log-gammas. Keep their f32 magnitudes modest and reserve several
    // rounding units before either atom's log probability can round to zero.
    // For x >= 2, x * ln(x) bounds log-gamma. For x < 1, the recurrence
    // bounds it by -ln(x). The absolute value stays below 1 on [1, 2].
    let largest = a + b + 1.0;
    let magnitude = (largest * largest.ln()).max(-a.min(b).ln()).max(1.0);
    let rounding_margin = 8.0 * f64::from(f32::EPSILON) * magnitude;
    magnitude <= 128.0 && p.min(1.0 - p) > rounding_margin
}

fn predictive_scale(prior: f64, noise: f64) -> Option<f64> {
    // The determinizer uses sqrt(pow(prior, 2) + pow(noise, 2)). Require
    // exact, normal f32 variances and addition before folding it to a literal.
    let a = prior.powi(2);
    let b = noise.powi(2);
    let total = a + b;
    if [a, b, total].into_iter().any(|var| {
        let narrow = var as f32;
        !narrow.is_normal() || f64::from(narrow) != var
    }) {
        return None;
    }
    let scale = total.sqrt();
    (scale as f32 == (total as f32).sqrt()).then_some(scale)
}

pub(super) fn seed(
    m: &Module,
    id: NodeId,
    term: usize,
    resolve: impl Fn(NodeId) -> Option<NodeId>,
    facts: &mut Vec<Fact>,
) {
    if let Some(fact) = predictive(m, id, term, resolve) {
        facts.push(fact);
    }
}

fn predictive(
    m: &Module,
    id: NodeId,
    term: usize,
    resolve: impl Fn(NodeId) -> Option<NodeId>,
) -> Option<Fact> {
    let &[measure, point] =
        positional(m, id, "logdensityof").or_else(|| positional(m, id, "densityof"))?
    else {
        return None;
    };
    let measure = resolve(measure)?;
    if m.phase_of(measure) != Some(Phase::Fixed) {
        return None;
    }
    let &[prior, kernel] = positional(m, measure, "kchain")? else {
        return None;
    };
    let prior = resolve(prior)?;
    let kernel = resolve(kernel)?;
    if !normalized_scalar(m, prior, ScalarType::Real)
        || m.is_sample_result(prior)
        || m.is_sample_result(kernel)
    {
        return None;
    }
    let reified = expect_builtin_call(m, kernel, "kernelof")?;
    let [body] = reified.args.as_ref() else {
        return None;
    };
    let Some(Inputs::Spec(inputs)) = &reified.inputs else {
        return None;
    };
    // Only a fresh local placeholder and an anonymous draw are admissible.
    // A named draw could carry a query's identity or another dependency.
    let [(_, latent)] = inputs.as_ref() else {
        return None;
    };
    if latent.ns != RefNs::Local || !reified.named.is_empty() {
        return None;
    }
    let &[likelihood] = positional(m, *body, "draw")? else {
        return None;
    };
    if m.is_sample_result(likelihood) {
        return None;
    }

    let binomial = expect_builtin_call(m, likelihood, "Binomial");
    let (domain, parameter, noise, prior_family, prior_parameters) =
        if let Some(normal) = expect_builtin_call(m, likelihood, "Normal") {
            let [mu, sigma] = parameters(m, normal, ["mu", "sigma"])?;
            (ScalarType::Real, mu, Some(sigma), "Normal", ["mu", "sigma"])
        } else if let Some(bernoulli) = expect_builtin_call(m, likelihood, "Bernoulli") {
            let [p] = parameters(m, bernoulli, ["p"])?;
            (ScalarType::Integer, p, None, "Beta", ["alpha", "beta"])
        } else if let Some(binomial) = binomial {
            let [trials, p] = parameters(m, binomial, ["n", "p"])?;
            let trials = resolve(trials)?;
            if !matches!(m.node(trials), Node::Lit(Scalar::Int(1)))
                || m.phase_of(trials) != Some(Phase::Fixed)
                || m.is_sample_result(trials)
                || m.sample_origin(trials).is_some()
            {
                return None;
            }
            binary_point(m, point, &resolve)?;
            (ScalarType::Integer, p, None, "Beta", ["alpha", "beta"])
        } else {
            let poisson = expect_builtin_call(m, likelihood, "Poisson")?;
            let [rate] = parameters(m, poisson, ["rate"])?;
            (ScalarType::Integer, rate, None, "Gamma", ["shape", "rate"])
        };
    if !normalized_scalar(m, likelihood, domain)
        || !normalized_scalar(m, measure, domain)
        || !matches!(m.node(parameter), Node::Ref(reference) if reference == latent)
    {
        return None;
    }
    let [a, b] = parameters(
        m,
        expect_builtin_call(m, prior, prior_family)?,
        prior_parameters,
    )?;
    let a = fixed_real(m, a, &resolve)?;
    let b = positive_normal(fixed_real(m, b, &resolve)?)?;
    use FactArg::{Expr, Int};
    if let Some(sigma) = noise {
        let sigma = positive_normal(fixed_real(m, sigma, &resolve)?)?;
        let scale = predictive_scale(b, sigma)?;
        return Some(Fact {
            name: "conjugate-normal-mean",
            args: vec![
                Expr(term),
                Int(a.to_bits() as i64),
                Int(scale.to_bits() as i64),
            ],
        });
    }
    let a = positive_normal(a)?;
    if prior_family == "Gamma" {
        return Some(Fact {
            name: "conjugate-gamma-poisson",
            args: vec![Expr(term), Int(a.to_bits() as i64), Int(b.to_bits() as i64)],
        });
    }
    // Match the scaled calculation used by the native Beta–Bernoulli path.
    // Leave close-to-endpoint shapes and cross-precision rounding differences
    // to its existing guarded lowering.
    let scale = a.max(b);
    let p = (a / scale) / ((a / scale) + (b / scale));
    if binomial.is_some() && !well_conditioned_binomial(a, b, p) {
        return None;
    }
    let a32 = (a as f32) / (scale as f32);
    let b32 = (b as f32) / (scale as f32);
    let p32 = a32 / (a32 + b32);
    ((f32::MIN_POSITIVE..1.0).contains(&p32) && p as f32 == p32).then_some(Fact {
        name: "conjugate-unit-predictive",
        args: vec![Expr(term), Int(p.to_bits() as i64)],
    })
}
