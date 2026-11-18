//! Gaussian predictives from the observation mean and centered RMS.

use super::{Data, Evidence, Update, UpdatedLaw, log_beta, op, real, safe};
use crate::posterior::map;
use flatppl_core::{Module, NodeId};

const LOG_2PI: f64 = 1.8378770664093453;

fn square_ratio(m: &mut Module, value: NodeId, scale: NodeId) -> NodeId {
    let ratio = op(m, "divide", &[value, scale]);
    op(m, "mul", &[ratio, ratio])
}

struct Moments {
    origin: NodeId,
    offset: NodeId,
    rms: NodeId,
    finite: NodeId,
}

impl Moments {
    fn difference(&self, m: &mut Module, mean: NodeId) -> NodeId {
        let delta = op(m, "sub", &[self.origin, mean]);
        op(m, "add", &[delta, self.offset])
    }
}

/// Keep the mean as an origin and a small offset. Scaling the offsets before
/// centering and squaring also avoids overflow for widely separated data.
fn moments(m: &mut Module, plan: &mut Evidence, data: Data) -> Moments {
    let y = data.apply(m, "real", &[data.value]);
    let finite = data.apply(m, "isfinite", &[y]);
    let zero = real(m, 0.);
    let clean = data.apply(m, "ifelse", &[finite, y, zero]);
    if data.count.is_none() {
        return Moments {
            origin: clean,
            offset: zero,
            rms: zero,
            finite,
        };
    }
    let finite = op(m, "lall", &[finite]);
    let finite = plan.summary(m, finite);
    // Project zero onto the data interval. All differences from this origin
    // fit the observation dtype, including for opposite-signed extremes.
    let low = op(m, "minimum", &[clean]);
    let high = op(m, "maximum", &[clean]);
    let clipped = op(m, "min", &[zero, high]);
    let origin = op(m, "max", &[low, clipped]);
    let origin = plan.summary(m, origin);
    let offsets = map(m, "sub", &[clean, origin]);
    let abs = map(m, "abs", &[offsets]);
    let scale = op(m, "maximum", &[abs]);
    let positive = op(m, "gt", &[scale, zero]);
    let divisor = safe(m, positive, scale, 1.);
    let z = map(m, "divide", &[offsets, divisor]);
    let center = op(m, "mean", &[z]);
    let offset = op(m, "mul", &[scale, center]);
    let offset = plan.summary(m, offset);
    let deviations = map(m, "sub", &[z, center]);
    let z2 = map(m, "mul", &[deviations, deviations]);
    let mean_square = op(m, "mean", &[z2]);
    let mean_square = safe(m, positive, mean_square, 1.);
    let root = op(m, "sqrt", &[mean_square]);
    let rms = op(m, "mul", &[scale, root]);
    let rms = plan.summary(m, rms);
    Moments {
        origin,
        offset,
        rms,
        finite,
    }
}

pub(super) fn factor(
    m: &mut Module,
    plan: &mut Evidence,
    law: &mut UpdatedLaw,
    row: Update,
    data: Data,
    valid: NodeId,
) -> Option<(NodeId, NodeId)> {
    let moments = moments(m, plan, data);
    let rms = moments.rms;
    let n = data.count(m);
    let half = real(m, 0.5);
    let q = op(m, "mul", &[n, half]);
    let log_2pi = real(m, LOG_2PI);
    let constant = op(m, "mul", &[q, log_2pi]);
    let score = match (*law, row) {
        (UpdatedLaw::Normal(mean, tau), Update::NormalMean(sigma)) => {
            let mean = safe(m, valid, mean, 0.);
            let tau = safe(m, valid, tau, 1.);
            let sigma = safe(m, valid, sigma, 1.);
            let scale = op(m, "max", &[sigma, tau]);
            let s2 = square_ratio(m, sigma, scale);
            let t2 = square_ratio(m, tau, scale);
            let nt2 = op(m, "mul", &[n, t2]);
            let h2 = op(m, "add", &[s2, nt2]);
            let log_scale = op(m, "log", &[scale]);
            let log_h2 = op(m, "log", &[h2]);
            let half_log_h2 = op(m, "mul", &[half, log_h2]);
            let log_h = op(m, "add", &[log_scale, half_log_h2]);
            let delta = moments.difference(m, mean);
            let between = square_ratio(m, delta, scale);
            let between = op(m, "divide", &[between, h2]);
            let within = square_ratio(m, rms, sigma);
            let quadratic = op(m, "add", &[within, between]);
            let quadratic = op(m, "mul", &[q, quadratic]);
            let rest = real(m, f64::from(data.count.unwrap_or(1) - 1));
            let log_s = op(m, "log", &[sigma]);
            let rest = op(m, "mul", &[rest, log_s]);
            let normalizer = op(m, "add", &[constant, rest]);
            let normalizer = op(m, "add", &[normalizer, log_h]);
            let energy = op(m, "add", &[normalizer, quadratic]);

            // h² = σ²+nτ². These weights and the posterior scale never square
            // either dimensional input directly.
            let w0 = op(m, "divide", &[s2, h2]);
            let wy = op(m, "divide", &[nt2, h2]);
            let m0 = op(m, "mul", &[w0, mean]);
            let mean_y = op(m, "add", &[moments.origin, moments.offset]);
            let my = op(m, "mul", &[wy, mean_y]);
            let mean = op(m, "add", &[m0, my]);
            let small = op(m, "min", &[sigma, tau]);
            let h = op(m, "sqrt", &[h2]);
            let posterior_sigma = op(m, "divide", &[small, h]);
            *law = UpdatedLaw::Normal(mean, posterior_sigma);
            op(m, "neg", &[energy])
        }
        (UpdatedLaw::InverseGamma(a, b), Update::NormalVariance(mu)) => {
            let a = safe(m, valid, a, 1.);
            let b = safe(m, valid, b, 1.);
            let mu = safe(m, valid, mu, 0.);
            let delta = moments.difference(m, mu);
            let abs_delta = op(m, "abs", &[delta]);
            let scale = op(m, "max", &[abs_delta, rms]);
            let zero = real(m, 0.);
            let positive = op(m, "gt", &[scale, zero]);
            let divisor = safe(m, positive, scale, 1.);
            let d2 = square_ratio(m, delta, divisor);
            let r2 = square_ratio(m, rms, divisor);
            let normalized = op(m, "add", &[d2, r2]);
            let normalized = op(m, "mul", &[q, normalized]);
            let root_b = op(m, "sqrt", &[b]);
            let large = op(m, "gt", &[scale, root_b]);
            let small_scale = op(m, "ifelse", &[large, zero, scale]);
            let ratio = square_ratio(m, small_scale, root_b);
            let ratio = op(m, "mul", &[ratio, normalized]);
            let small_log = op(m, "log1p", &[ratio]);
            // When S/b is large, log(1+S/b) = 2 log(d/√b) + log(S/d²+b/d²).
            let big_scale = op(m, "ifelse", &[large, scale, root_b]);
            let ratio = square_ratio(m, root_b, big_scale);
            let inside = op(m, "add", &[normalized, ratio]);
            let inside = op(m, "log", &[inside]);
            let log_scale = op(m, "log", &[big_scale]);
            let log_b = op(m, "log", &[b]);
            let twice = op(m, "add", &[log_scale, log_scale]);
            let large_log = op(m, "sub", &[twice, log_b]);
            let large_log = op(m, "add", &[large_log, inside]);
            let increase = op(m, "ifelse", &[large, large_log, small_log]);
            let next_a = op(m, "add", &[a, q]);
            let energy = op(m, "mul", &[next_a, increase]);
            let scale_term = op(m, "mul", &[q, log_b]);
            let function = plan.logbeta(m);
            let beta = log_beta(m, function, a, q, false);
            let lgq = op(m, "loggamma", &[q]);
            let ratio = op(m, "sub", &[lgq, beta]);
            let score = op(m, "sub", &[ratio, scale_term]);
            let score = op(m, "sub", &[score, constant]);
            let sum = op(m, "mul", &[normalized, scale]);
            let sum = op(m, "mul", &[sum, scale]);
            let next_b = op(m, "add", &[b, sum]);
            *law = UpdatedLaw::InverseGamma(next_a, next_b);
            op(m, "sub", &[score, energy])
        }
        _ => return None,
    };
    Some((score, moments.finite))
}
