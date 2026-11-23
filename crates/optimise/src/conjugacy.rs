//! Evidence for closed conjugate predictive density queries.

use std::collections::{BTreeSet, HashSet};

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

fn supported_outcomes(
    m: &Module,
    point: NodeId,
    trials: i64,
    resolve: &impl Fn(NodeId) -> Option<NodeId>,
) -> Option<BTreeSet<i64>> {
    // The generic beta-binomial log-gamma expression can have poles beyond
    // its support. Prove each outcome without trusting a runtime input's
    // nominal value set or erasing its authored condition.
    let mut pending = vec![point];
    let mut seen = HashSet::new();
    let mut outcomes = BTreeSet::new();
    while let Some(id) = pending.pop() {
        let id = resolve(id)?;
        if !seen.insert(id) {
            continue;
        }
        if seen.len() > 32 || m.is_sample_result(id) || m.sample_origin(id).is_some() {
            return None;
        }
        if let Node::Lit(Scalar::Int(value)) = m.node(id) {
            if !(0..=trials).contains(value) {
                return None;
            }
            outcomes.insert(*value);
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
    (!outcomes.is_empty()).then_some(outcomes)
}

fn binomial_rounding_margin(a: f64, b: f64, trials: i64) -> Option<f64> {
    // The binomial coefficient and two log-beta terms are differences of
    // log-gammas. Keep their f32 magnitudes modest and reserve rounding
    // margin before an atom's log probability can round to zero.
    // For x >= 2, x * ln(x) bounds log-gamma. For x < 1, the recurrence
    // bounds it by -ln(x). The absolute value stays below 1 on [1, 2].
    let trials = trials as f64;
    let largest = (a + b + trials).max(trials + 1.0);
    let magnitude = (largest * largest.ln()).max(-a.min(b).ln()).max(1.0);
    (magnitude <= 128.0).then_some(8.0 * f64::from(f32::EPSILON) * magnitude)
}

fn beta_binomial_masses(
    m: &Module,
    point: NodeId,
    trials: i64,
    a: f64,
    b: f64,
    resolve: &impl Fn(NodeId) -> Option<NodeId>,
) -> Option<FactArg> {
    let margin = binomial_rounding_margin(a, b, trials)?;
    let outcomes = supported_outcomes(m, point, trials, resolve)?;
    // Each additional outcome contributes one test and one select. A lone
    // atom may replace a literal, while an authored conditional must still run.
    if outcomes.len() > 8
        || (outcomes.len() == 1 && !matches!(m.node(resolve(point)?), Node::Lit(Scalar::Int(_))))
    {
        return None;
    }
    // P(0) = (b)_n / (a + b)_n. Each successive mass follows from
    // P(k+1) / P(k) = (n-k)(a+k) / ((k+1)(b+n-k-1)).
    let mut mass = (0..trials)
        .map(|i| (b + i as f64) / (a + b + i as f64))
        .product::<f64>();
    let mut selected = Vec::with_capacity(outcomes.len());
    let last = *outcomes.last()?;
    for k in 0..=last {
        if outcomes.contains(&k) {
            let log_mass = mass.ln();
            if !mass.is_finite() || mass < f64::from(f32::MIN_POSITIVE) || log_mass >= -margin {
                return None;
            }
            selected.push((k, (-log_mass).to_bits() as i64));
        }
        if k == last {
            break;
        }
        mass *= ((trials - k) as f64 * (a + k as f64))
            / ((k + 1) as f64 * (b + (trials - k - 1) as f64));
    }
    use FactArg::{Call, Int};
    let (last, preceding) = selected.split_last()?;
    let tail = Call("ConjugateLastMass", vec![Int(last.1)]);
    Some(preceding.iter().rev().fold(tail, |tail, &(k, log)| {
        Call("ConjugateMass", vec![Int(k), Int(log), tail])
    }))
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

fn laplace_scale(rate: f64) -> Option<f64> {
    let twice = 2.0 * rate;
    let narrow = twice as f32;
    let scale = twice.sqrt().recip();
    (narrow.is_normal() && f64::from(narrow) == twice && scale as f32 == narrow.sqrt().recip())
        .then_some(scale)
}

fn student_parameters(shape: f64, beta: f64) -> Option<(f64, f64, f64)> {
    let nu = 2.0 * shape;
    let ratio = beta / shape;
    let scale = ratio.sqrt();
    let narrow = (beta as f32) / (shape as f32);
    // The normalizer has a finite recurrence for integer degrees of freedom.
    // Bound its work and preserve the original standardized point in f32.
    let degrees = nu as i64;
    if nu != degrees as f64
        || !(1..=64).contains(&degrees)
        || !narrow.is_normal()
        || scale as f32 != narrow.sqrt()
    {
        return None;
    }
    // B(1/2, 1/2) = pi, B(1, 1/2) = 2 and B(x+1, 1/2) = B(x, 1/2)*x/(x+1/2).
    let (first, base) = if degrees % 2 == 0 {
        (1.0, 2.0)
    } else {
        (0.5, std::f64::consts::PI)
    };
    let log_beta = (0..(degrees - 1) / 2).fold(base.ln(), |sum, step| {
        let x = first + step as f64;
        sum + (x / (x + 0.5)).ln()
    });
    let norm = scale.ln() + nu.sqrt().ln() + log_beta;
    Some((nu, scale, norm))
}

fn variance_predictive(
    m: &Module,
    term: usize,
    prior: NodeId,
    location: NodeId,
    resolve: &impl Fn(NodeId) -> Option<NodeId>,
) -> Option<Fact> {
    use FactArg::{Expr, Int};
    let location = fixed_real(m, location, resolve)?.to_bits() as i64;
    if let Some(exponential) = expect_builtin_call(m, prior, "Exponential") {
        let [rate] = parameters(m, exponential, ["rate"])?;
        let rate = positive_normal(fixed_real(m, rate, resolve)?)?;
        return Some(Fact {
            name: "conjugate-exponential-variance",
            args: vec![
                Expr(term),
                Int(location),
                Int(laplace_scale(rate)?.to_bits() as i64),
            ],
        });
    }
    let inverse_gamma = expect_builtin_call(m, prior, "InverseGamma")?;
    let [shape, beta] = parameters(m, inverse_gamma, ["shape", "scale"])?;
    let shape = positive_normal(fixed_real(m, shape, resolve)?)?;
    let beta = positive_normal(fixed_real(m, beta, resolve)?)?;
    let (nu, scale, norm) = student_parameters(shape, beta)?;
    Some(Fact {
        name: "conjugate-inverse-gamma-variance",
        args: vec![
            Expr(term),
            Int(location),
            Int(scale.to_bits() as i64),
            Int(nu.to_bits() as i64),
            Int(((nu + 1.0) / 2.0).to_bits() as i64),
            Int(norm.to_bits() as i64),
        ],
    })
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

    use FactArg::{Expr, Int};
    let direct = |id| matches!(m.node(id), Node::Ref(reference) if reference == latent);
    let scalar =
        |domain| normalized_scalar(m, likelihood, domain) && normalized_scalar(m, measure, domain);
    if let Some(normal) = expect_builtin_call(m, likelihood, "Normal") {
        if !scalar(ScalarType::Real) {
            return None;
        }
        let [mu, sigma] = parameters(m, normal, ["mu", "sigma"])?;
        if !direct(mu) {
            let &[variance] = positional(m, resolve(sigma)?, "sqrt")? else {
                return None;
            };
            return direct(variance)
                .then(|| variance_predictive(m, term, prior, mu, &resolve))
                .flatten();
        }
        let [location, scale] =
            parameters(m, expect_builtin_call(m, prior, "Normal")?, ["mu", "sigma"])?;
        let location = fixed_real(m, location, &resolve)?;
        let scale = positive_normal(fixed_real(m, scale, &resolve)?)?;
        let sigma = positive_normal(fixed_real(m, sigma, &resolve)?)?;
        let scale = predictive_scale(scale, sigma)?;
        return Some(Fact {
            name: "conjugate-normal-mean",
            args: vec![
                Expr(term),
                Int(location.to_bits() as i64),
                Int(scale.to_bits() as i64),
            ],
        });
    }
    let (parameter, trials, prior_family, prior_parameters) =
        if let Some(binomial) = expect_builtin_call(m, likelihood, "Binomial") {
            let [trials, p] = parameters(m, binomial, ["n", "p"])?;
            let trials = resolve(trials)?;
            let Node::Lit(Scalar::Int(n @ 1..)) = m.node(trials) else {
                return None;
            };
            if m.phase_of(trials) != Some(Phase::Fixed)
                || m.is_sample_result(trials)
                || m.sample_origin(trials).is_some()
            {
                return None;
            }
            (p, Some(*n), "Beta", ["alpha", "beta"])
        } else if let Some(bernoulli) = expect_builtin_call(m, likelihood, "Bernoulli") {
            let [p] = parameters(m, bernoulli, ["p"])?;
            (p, None, "Beta", ["alpha", "beta"])
        } else {
            let poisson = expect_builtin_call(m, likelihood, "Poisson")?;
            let [rate] = parameters(m, poisson, ["rate"])?;
            (rate, None, "Gamma", ["shape", "rate"])
        };
    if !scalar(ScalarType::Integer) || !direct(parameter) {
        return None;
    }
    let [a, b] = parameters(
        m,
        expect_builtin_call(m, prior, prior_family)?,
        prior_parameters,
    )?;
    let a = positive_normal(fixed_real(m, a, &resolve)?)?;
    let b = positive_normal(fixed_real(m, b, &resolve)?)?;
    if prior_family == "Gamma" {
        return Some(Fact {
            name: "conjugate-gamma-poisson",
            args: vec![Expr(term), Int(a.to_bits() as i64), Int(b.to_bits() as i64)],
        });
    }
    if let Some(trials @ 2..) = trials {
        return Some(Fact {
            name: "conjugate-beta-binomial",
            args: vec![
                Expr(term),
                beta_binomial_masses(m, point, trials, a, b, &resolve)?,
            ],
        });
    }
    // Match the scaled calculation used by the native Beta–Bernoulli path.
    // Leave close-to-endpoint shapes and cross-precision rounding differences
    // to its existing guarded lowering.
    let scale = a.max(b);
    let p = (a / scale) / ((a / scale) + (b / scale));
    if let Some(trials) = trials {
        supported_outcomes(m, point, trials, &resolve)?;
        if p.min(1.0 - p) <= binomial_rounding_margin(a, b, trials)? {
            return None;
        }
    }
    let a32 = (a as f32) / (scale as f32);
    let b32 = (b as f32) / (scale as f32);
    let p32 = a32 / (a32 + b32);
    ((f32::MIN_POSITIVE..1.0).contains(&p32) && p as f32 == p32).then_some(Fact {
        name: "conjugate-unit-predictive",
        args: vec![Expr(term), Int(p.to_bits() as i64)],
    })
}
