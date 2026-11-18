//! Beta updates from ordered Bernoulli observations and Binomial row counts.

use super::counts::allocation_poisson;
use super::{Data, UpdatePlan, UpdatedLaw, log_beta, op, real, safe};
use crate::posterior::{supported_counts, valid_trials};
use crate::sufficient_statistics::data_only;
use flatppl_core::{Module, Node, NodeId, Scalar};
use std::collections::HashSet;

/// A live trial count sets an upper bound on every row. Keep the maximum in
/// the source count type so adjacent integers beyond the real mantissa remain distinct.
fn largest(m: &mut Module, data: Data, values: NodeId, rows: u32) -> NodeId {
    if data.integer {
        let maxima = op(m, "cummax", &[values]);
        let last = m.alloc(Node::Lit(Scalar::Int(rows.into())));
        op(m, "get", &[maxima, last])
    } else {
        op(m, "maximum", &[values])
    }
}

pub(super) fn factor(
    m: &mut Module,
    plan: &mut UpdatePlan,
    law: &mut UpdatedLaw,
    shapes: [NodeId; 2],
    trials: Option<NodeId>,
    data: Data,
    valid: NodeId,
) -> (NodeId, NodeId) {
    let n = trials.unwrap_or_else(|| m.alloc(Node::Lit(Scalar::Int(1))));
    let live_trials = data
        .count
        .filter(|_| trials.is_some() && !data_only(m, n, &HashSet::new(), &plan.boundary));
    let upper = live_trials.is_none().then_some(n);
    let supported = supported_counts(m, data.value, data.integer, upper, data.count.is_some());
    let supported = plan.summary(m, supported);
    let int_zero = m.alloc(Node::Lit(Scalar::Int(0)));
    let cells = data.apply(m, "ifelse", &[supported, data.value, int_zero]);
    let successes = data.apply(m, "real", &[cells]);
    let k = data.sum(m, successes);
    let k = plan.summary(m, k);

    let (observed, f) = if let Some(rows) = live_trials {
        let maximum = largest(m, data, cells, rows);
        let maximum = plan.summary(m, maximum);
        let in_range = op(m, "le", &[maximum, n]);
        let observed = op(m, "land", &[supported, in_range]);
        // Sum failures as sum(max-y) + rows*(n-max). Both differences are
        // taken before conversion to real, retaining even a single failure.
        let gaps = data.apply(m, "sub", &[maximum, cells]);
        let gaps = data.apply(m, "real", &[gaps]);
        let gaps = data.sum(m, gaps);
        let gaps = plan.summary(m, gaps);
        let remaining = op(m, "sub", &[n, maximum]);
        let remaining = op(m, "real", &[remaining]);
        let rows = data.count(m);
        let extra = op(m, "mul", &[rows, remaining]);
        let failures = op(m, "add", &[gaps, extra]);
        (observed, safe(m, observed, failures, 0.))
    } else {
        let failures = data.apply(m, "sub", &[n, cells]);
        let failures = data.apply(m, "ifelse", &[supported, failures, int_zero]);
        let failures = data.apply(m, "real", &[failures]);
        let failures = data.sum(m, failures);
        (supported, plan.summary(m, failures))
    };
    let a = safe(m, valid, shapes[0], 1.);
    let b = safe(m, valid, shapes[1], 1.);
    *law = UpdatedLaw::Beta(op(m, "add", &[a, k]), op(m, "add", &[b, f]));
    if !plan.with_evidence {
        return (real(m, 0.), observed);
    }
    if trials.is_none() && data.count.is_none() {
        return (bernoulli(m, k, a, b), observed);
    }
    let function = plan.logbeta(m);
    let pooled = pooled_log_mass(m, function, [a, b], k, f);
    if data.count.unwrap_or(1) == 1 {
        return (pooled, observed);
    }
    let correction = if trials.is_none() {
        // Distinguish this particular ordered sequence among those with k successes.
        let one = real(m, 1.);
        let kp = op(m, "add", &[k, one]);
        let fp = op(m, "add", &[f, one]);
        let beta = log_beta(m, function, kp, fp);
        let rows = data.count(m);
        let log_n1 = op(m, "log1p", &[rows]);
        op(m, "add", &[beta, log_n1])
    } else {
        let n_valid = valid_trials(m, n);
        let usable = op(m, "land", &[observed, n_valid]);
        let failures = data.apply(m, "sub", &[n, cells]);
        let failures = data.apply(m, "real", &[failures]);
        let zero = real(m, 0.);
        let failures = data.apply(m, "ifelse", &[usable, failures, zero]);
        let f = safe(m, usable, f, 0.);
        let row_size = op(m, "real", &[n]);
        let row_size = safe(m, usable, row_size, 0.);
        let rows = data.count(m);
        let correction = binomial_allocation(m, rows, row_size, [(successes, k), (failures, f)]);
        plan.summary(m, correction)
    };
    (op(m, "add", &[pooled, correction]), observed)
}

/// Conditional log probability of the observed two-column table at fixed
/// row and column totals. Centered Poisson masses avoid subtracting large,
/// nearly equal log binomial coefficients in balanced tables.
fn binomial_allocation(
    m: &mut Module,
    rows: NodeId,
    row_size: NodeId,
    columns: [(NodeId, NodeId); 2],
) -> NodeId {
    let [successes, failures] = columns.map(|(cells, total)| {
        let rate = op(m, "divide", &[total, rows]);
        let scores = allocation_poisson(m, rate, cells, true);
        let sum = op(m, "sum", &[scores]);
        let column = allocation_poisson(m, total, total, false);
        op(m, "sub", &[sum, column])
    });
    let row = allocation_poisson(m, row_size, row_size, false);
    let by_row = op(m, "mul", &[row, rows]);
    let total = op(m, "add", &[columns[0].1, columns[1].1]);
    let pooled = allocation_poisson(m, total, total, false);
    let margins = op(m, "sub", &[by_row, pooled]);
    let by_column = op(m, "add", &[successes, failures]);
    op(m, "sub", &[by_column, margins])
}

/// Predict one Bernoulli event using the prior mean or its complement.
fn bernoulli(m: &mut Module, k: NodeId, a: NodeId, b: NodeId) -> NodeId {
    let zero = real(m, 0.);
    let success = op(m, "gt", &[k, zero]);
    let observed = op(m, "ifelse", &[success, a, b]);
    let other = op(m, "ifelse", &[success, b, a]);
    let at_max = op(m, "ge", &[observed, other]);
    let larger = op(m, "ifelse", &[at_max, observed, other]);
    let smaller = op(m, "ifelse", &[at_max, other, observed]);
    let fraction = op(m, "divide", &[smaller, larger]);
    let correction = op(m, "log1p", &[fraction]);
    let positive = op(m, "gt", &[fraction, zero]);
    let positive_fraction = safe(m, positive, fraction, 1.);
    let log_fraction = op(m, "log", &[positive_fraction]);
    let log_observed = op(m, "log", &[observed]);
    let log_other = op(m, "log", &[other]);
    let difference = op(m, "sub", &[log_observed, log_other]);
    let log_fraction = op(m, "ifelse", &[positive, log_fraction, difference]);
    let relative = op(m, "ifelse", &[at_max, zero, log_fraction]);
    op(m, "sub", &[relative, correction])
}

/// log(j! / (x)_j) = log(j) + log B(x,j), with value zero at j=0.
/// This avoids subtracting two large log-gammas for a concentrated prior.
fn log_factorial_ratio(m: &mut Module, function: NodeId, x: NodeId, j: NodeId) -> NodeId {
    let zero = real(m, 0.);
    let positive = op(m, "gt", &[j, zero]);
    let safe = safe(m, positive, j, 1.);
    let log_j = op(m, "log", &[safe]);
    let beta = log_beta(m, function, x, safe);
    let value = op(m, "add", &[log_j, beta]);
    op(m, "ifelse", &[positive, value, zero])
}

fn pooled_log_mass(
    m: &mut Module,
    function: NodeId,
    [a, b]: [NodeId; 2],
    k: NodeId,
    f: NodeId,
) -> NodeId {
    let total = op(m, "add", &[k, f]);
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
    let beta = log_beta(m, function, half_n1, half);
    let log_pi = real(m, std::f64::consts::PI.ln());
    let correction = op(m, "sub", &[beta, log_pi]);
    let twice_h = op(m, "add", &[h, h]);
    let duplication = op(m, "add", &[twice_h, correction]);
    let denom = op(m, "ifelse", &[finite, h, duplication]);
    let kr = log_factorial_ratio(m, function, a, k);
    let fr = log_factorial_ratio(m, function, b, f);
    let score = op(m, "sub", &[denom, kr]);
    op(m, "sub", &[score, fr])
}
