//! Log evidence for Beta priors with ordered Bernoulli or Binomial observations.

use super::{Data, Update, UpdatedLaw, match_update, op, real};
use flatppl_core::graph::{build_record, special_functions_alias};
use flatppl_core::{Call, CallHead, Module, Node, NodeId, Ref, RefNs};

/// Log mass of an unnormalized Beta update. An impossible observation has zero mass.
pub fn beta_log_evidence(m: &mut Module, posterior: NodeId) -> Option<NodeId> {
    let mut sum = None;
    let mut logbeta = None;
    let matched = match_update(m, posterior, |m, prior, _, row, data, validity| {
        let (UpdatedLaw::Beta(a, b), Update::Binomial(trials)) = (prior, row) else {
            return None;
        };
        let valid = op(m, "land", &validity);
        let one = real(m, 1.0);
        let a = op(m, "ifelse", &[valid, a, one]);
        let b = op(m, "ifelse", &[valid, b, one]);
        let factor = factor_log_mass(m, &mut logbeta, [a, b], trials, data, valid);
        sum = Some(match sum {
            Some(previous) => op(m, "add", &[previous, factor]),
            None => factor,
        });
        Some(())
    })?;
    let minus_inf = real(m, f64::NEG_INFINITY);
    let log_mass = op(m, "ifelse", &[matched.supported, sum?, minus_inf]);
    // A unit-probability event adds zero on every valid model. Its native
    // probability check preserves the source parameter error on invalid models.
    let one = real(m, 1.0);
    let invalid = real(m, -1.0);
    let p = op(m, "ifelse", &[matched.model_valid, one, invalid]);
    let name = m.intern("p");
    let parameters = build_record(m, &[(name, p)]);
    let name = m.intern("Bernoulli");
    let kernel = m.alloc(Node::Const(name));
    let anchor = op(m, "builtin_logdensityof", &[kernel, parameters, one]);
    Some(op(m, "add", &[log_mass, anchor]))
}

fn log_beta(m: &mut Module, function: NodeId, x: NodeId, y: NodeId, array: bool) -> NodeId {
    if array {
        op(m, "broadcast", &[function, x, y])
    } else {
        m.alloc(Node::Call(Call {
            head: CallHead::User(function),
            args: Box::new([x, y]),
            named: Box::new([]),
            inputs: None,
        }))
    }
}

/// log(j! / (x)_j) = log(j) + log B(x,j), with value zero at j=0.
/// This avoids subtracting two large log-gammas for a concentrated prior.
fn log_factorial_ratio(m: &mut Module, function: NodeId, x: NodeId, j: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let positive = op(m, "gt", &[j, zero]);
    let safe = op(m, "ifelse", &[positive, j, one]);
    let log_j = op(m, "log", &[safe]);
    let beta = log_beta(m, function, x, safe, false);
    let value = op(m, "add", &[log_j, beta]);
    op(m, "ifelse", &[positive, value, zero])
}

/// log C(k+f,k) = -log(k+f+1) - log B(k+1,f+1).
fn log_choose(m: &mut Module, function: NodeId, data: Data, k: NodeId, f: NodeId) -> NodeId {
    let one = real(m, 1.0);
    let n = data.apply(m, "add", &[k, f]);
    let k1 = data.apply(m, "add", &[k, one]);
    let f1 = data.apply(m, "add", &[f, one]);
    let norm = log_beta(m, function, k1, f1, data.count.is_some());
    let log_n1 = data.apply(m, "log1p", &[n]);
    let sum = data.apply(m, "add", &[norm, log_n1]);
    data.apply(m, "neg", &[sum])
}

fn factor_log_mass(
    m: &mut Module,
    logbeta: &mut Option<NodeId>,
    shapes: [NodeId; 2],
    trials: Option<NodeId>,
    data: Data,
    valid: NodeId,
) -> NodeId {
    let zero = real(m, 0.0);
    if data.count == Some(0) {
        return zero;
    }
    let [a, b] = shapes;
    if trials.is_none() && data.count.is_none() {
        // One Bernoulli event needs only the prior's mean or its complement.
        let success = op(m, "gt", &[data.value, zero]);
        let observed = op(m, "ifelse", &[success, a, b]);
        let other = op(m, "ifelse", &[success, b, a]);
        let at_max = op(m, "ge", &[observed, other]);
        let larger = op(m, "ifelse", &[at_max, observed, other]);
        let smaller = op(m, "ifelse", &[at_max, other, observed]);
        let fraction = op(m, "divide", &[smaller, larger]);
        let correction = op(m, "log1p", &[fraction]);
        let positive = op(m, "gt", &[fraction, zero]);
        let one = real(m, 1.0);
        let safe = op(m, "ifelse", &[positive, fraction, one]);
        let log_fraction = op(m, "log", &[safe]);
        let log_observed = op(m, "log", &[observed]);
        let log_other = op(m, "log", &[other]);
        let difference = op(m, "sub", &[log_observed, log_other]);
        let log_fraction = op(m, "ifelse", &[positive, log_fraction, difference]);
        let relative = op(m, "ifelse", &[at_max, zero, log_fraction]);
        return op(m, "sub", &[relative, correction]);
    }
    let function = *logbeta.get_or_insert_with(|| {
        let alias = special_functions_alias(m);
        let name = m.intern("logbeta");
        m.alloc(Node::Ref(Ref {
            ns: RefNs::Module(alias),
            name,
        }))
    });
    let n = trials.unwrap_or_else(|| m.alloc(Node::Lit(flatppl_core::Scalar::Int(1))));
    let failures = data.apply(m, "sub", &[n, data.value]);
    let failures = data.apply(m, "real", &[failures]);
    let successes = data.apply(m, "real", &[data.value]);
    let k = data.apply(m, "ifelse", &[valid, successes, zero]);
    let f = data.apply(m, "ifelse", &[valid, failures, zero]);
    let total_k = data.sum(m, k);
    let total_f = data.sum(m, f);
    let total = op(m, "add", &[total_k, total_f]);
    let ab = op(m, "add", &[a, b]);
    let finite = op(m, "isfinite", &[ab]);
    let half = real(m, 0.5);
    // Form the midpoint without a common factor that a backend could pull
    // outside the sum and overflow before halving.
    let lo = op(m, "min", &[a, b]);
    let hi = op(m, "max", &[a, b]);
    let distance = op(m, "sub", &[hi, lo]);
    let half_distance = op(m, "mul", &[distance, half]);
    let half_ab = op(m, "add", &[lo, half_distance]);
    let x = op(m, "ifelse", &[finite, ab, half_ab]);
    let half_total = op(m, "mul", &[total, half]);
    let j = op(m, "ifelse", &[finite, total, half_total]);
    let h = log_factorial_ratio(m, function, x, j);
    // Gamma duplication evaluates h(a+b,N) when the sum exceeds the real dtype.
    // At that scale (a+b)/2 and (a+b+1)/2 are the same floating-point value.
    let half_n1 = op(m, "add", &[half_total, half]);
    let beta = log_beta(m, function, half_n1, half, false);
    let log_pi = real(m, std::f64::consts::PI.ln());
    let correction = op(m, "sub", &[beta, log_pi]);
    let twice_h = op(m, "add", &[h, h]);
    let duplication = op(m, "add", &[twice_h, correction]);
    let denom = op(m, "ifelse", &[finite, h, duplication]);
    let kr = log_factorial_ratio(m, function, a, total_k);
    let fr = log_factorial_ratio(m, function, b, total_f);
    let score = op(m, "sub", &[denom, kr]);
    let score = op(m, "sub", &[score, fr]);
    if data.count.unwrap_or(1) == 1 {
        return score;
    }
    // The pooled Beta–Binomial score counts all allocations of the successes.
    // Keep only arrangements consistent with the observations in this IID block.
    let pooled = log_choose(
        m,
        function,
        Data {
            count: None,
            ..data
        },
        total_k,
        total_f,
    );
    let correction = if trials.is_some() {
        let coefficients = log_choose(m, function, data, k, f);
        let sum = data.sum(m, coefficients);
        op(m, "sub", &[sum, pooled])
    } else {
        op(m, "neg", &[pooled])
    };
    op(m, "add", &[score, correction])
}
