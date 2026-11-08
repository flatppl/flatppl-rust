//! `log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)` without the cancellation of
//! three large log-gammas. For arguments of 8 and up, Stirling's series splits each
//! log-gamma into terms that cancel analytically and a small correction, so the
//! difference keeps its precision in the value and in its derivative. Every piece is
//! an ordinary op, so reverse mode differentiates the stable form.
//!
//! The split follows DiDonato and Morris, "Significant digit computation of the
//! incomplete beta function ratios", ACM TOMS 18 (1992), algorithm 708.

use super::*;

/// Below this, the Stirling correction loses accuracy and log-gamma is used directly.
const STIRLING_MIN: f64 = 8.0;

/// `lgamma(x) - [(x - 1/2) log x - x + log(2 pi) / 2]` for `x >= 8`, as a
/// polynomial in `1/x^2` (algorithm 708's `bcorr` coefficients).
const CORRECTION: [f64; 6] = [
    0.833333333333333e-01,
    -0.277777777760991e-02,
    0.793650666825390e-03,
    -0.595202931351870e-03,
    0.837308034031215e-03,
    -0.165322962780713e-02,
];

impl Emitter<'_> {
    fn log_gamma_correction(&mut self, x: &Value) -> Value {
        let one = self.scalar(1.0);
        let inverse = self.div(&one, x);
        let inverse_squared = self.mul(&inverse, &inverse);
        let (last, rest) = CORRECTION.split_last().expect("nonempty coefficients");
        let mut acc = self.scalar(*last);
        for &c in rest.iter().rev() {
            let scaled = self.mul(&acc, &inverse_squared);
            let coefficient = self.scalar(c);
            acc = self.add(&scaled, &coefficient);
        }
        self.mul(&acc, &inverse)
    }

    /// `r(x) = lgamma(x) - [(x - 1/2) log x - x + log(2 pi) / 2]` for `x > 0`: the
    /// series for `x >= 8`, otherwise `lgamma` itself, where nothing large cancels.
    fn stirling_remainder(&mut self, x: &Value) -> Value {
        let threshold = self.constant_like(STIRLING_MIN, x);
        let large = self.compare("GE", x, &threshold);
        let x_big = self.select(&large, x, &threshold);
        let series = self.log_gamma_correction(&x_big);
        let lgamma_x = self.lgamma(x);
        let half = self.scalar(0.5);
        let x_minus_half = self.sub(x, &half);
        let log_x = self.log(x);
        let t = self.mul(&x_minus_half, &log_x);
        let direct = self.sub(&lgamma_x, &t);
        let direct = self.add(&direct, x);
        let half_log_two_pi = self.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
        let direct = self.sub(&direct, &half_log_two_pi);
        self.select(&large, &series, &direct)
    }

    /// `sum(lgamma(alpha)) - lgamma(sum(alpha))` over the last axis of `alpha`, all
    /// entries positive. With `s = sum(alpha)` and `k` entries, Stirling's form
    /// cancels the `-alpha_i` terms against `+s` exactly:
    /// `sum(r(alpha_i)) - r(s) + sum((alpha_i - 1/2) log(alpha_i / s)) - (k - 1)/2 *
    /// (log s - log(2 pi))`. No log-gamma of `s` remains to cancel.
    pub(crate) fn log_multivariate_beta(&mut self, alpha: &Value) -> Value {
        let alpha = self.convert(alpha, ElemKind::Real);
        let total = self.reduce_sum(&alpha);
        let remainders = self.stirling_remainder(&alpha);
        let sum_remainders = self.reduce_sum(&remainders);
        let total_remainder = self.stirling_remainder(&total);
        let half = self.scalar(0.5);
        let share = self.div(&alpha, &total);
        let log_share = self.log(&share);
        let alpha_minus_half = self.sub(&alpha, &half);
        let weighted = self.mul(&alpha_minus_half, &log_share);
        let sum_weighted = self.reduce_sum(&weighted);
        // (k - 1) / 2 as a sum of halves, so the length need not be static.
        let halves = self.constant_like(0.5, &alpha);
        let half_k = self.reduce_sum(&halves);
        let half_k_minus_one = self.sub(&half_k, &half);
        let log_total = self.log(&total);
        let log_two_pi = self.scalar((2.0 * std::f64::consts::PI).ln());
        let log_ratio = self.sub(&log_total, &log_two_pi);
        let tail = self.mul(&half_k_minus_one, &log_ratio);
        let t = self.sub(&sum_remainders, &total_remainder);
        let t = self.add(&t, &sum_weighted);
        self.sub(&t, &tail)
    }

    /// `log B(a, b)` for positive `a` and `b`, elementwise.
    ///
    /// One comparison orders the pair, so tied arguments each keep their own
    /// derivative. Each branch reads arguments clamped into its own domain, so the
    /// branches `select` discards stay finite in reverse mode.
    pub(crate) fn log_beta(&mut self, a: &Value, b: &Value) -> Value {
        let a = self.convert(a, ElemKind::Real);
        let b = self.convert(b, ElemKind::Real);
        let (a, b) = self.broadcast_pair(&a, &b);
        let ordered = self.compare("LE", &a, &b);
        let x = self.select(&ordered, &a, &b);
        let y = self.select(&ordered, &b, &a);

        let threshold = self.constant_like(STIRLING_MIN, &x);
        let x_large = self.compare("GE", &x, &threshold);
        let y_large = self.compare("GE", &y, &threshold);
        let x_big = self.select(&x_large, &x, &threshold);
        let y_big = self.select(&y_large, &y, &threshold);
        let half = self.scalar(0.5);

        // lgamma(y) - lgamma(x + y) for y >= 8, with the Stirling terms cancelled:
        // corr(y) - corr(x + y) - (x + y - 1/2) log1p(x / y) - x log y + x.
        let difference_big_y = |e: &mut Self, x: &Value| {
            let sum = e.add(x, &y_big);
            let corr_y = e.log_gamma_correction(&y_big);
            let corr_sum = e.log_gamma_correction(&sum);
            let correction = e.sub(&corr_y, &corr_sum);
            let ratio = e.div(x, &y_big);
            let log1p_ratio = e.log1p(&ratio);
            let sum_minus_half = e.sub(&sum, &half);
            let t1 = e.mul(&sum_minus_half, &log1p_ratio);
            let log_y = e.log(&y_big);
            let t2 = e.mul(x, &log_y);
            let t = e.add(&t1, &t2);
            let t = e.sub(x, &t);
            e.add(&correction, &t)
        };

        // Both arguments 8 or more: lgamma(x) in Stirling form plus the difference.
        // The -x and +x cancel, and the logarithms regroup as
        // (x - 1/2) log(x / (x + y)) - y log1p(x / y) - log(y) / 2.
        let both_large = {
            let sum = self.add(&x_big, &y_big);
            let corr_x = self.log_gamma_correction(&x_big);
            let corr_y = self.log_gamma_correction(&y_big);
            let corr_sum = self.log_gamma_correction(&sum);
            let share = self.div(&x_big, &sum);
            let log_share = self.log(&share);
            let x_minus_half = self.sub(&x_big, &half);
            let t1 = self.mul(&x_minus_half, &log_share);
            let log_y = self.log(&y_big);
            let half_log_y = self.mul(&half, &log_y);
            let half_log_two_pi = self.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
            let ratio = self.div(&x_big, &y_big);
            let log1p_ratio = self.log1p(&ratio);
            let y_log1p = self.mul(&y_big, &log1p_ratio);
            let t = self.sub(&half_log_two_pi, &half_log_y);
            let t = self.add(&t, &corr_x);
            let t = self.add(&t, &corr_y);
            let t = self.sub(&t, &corr_sum);
            let t = self.add(&t, &t1);
            self.sub(&t, &y_log1p)
        };

        let one_large = {
            let lgamma_x = self.lgamma(&x);
            let difference = difference_big_y(self, &x);
            self.add(&lgamma_x, &difference)
        };

        let both_small = {
            let lgamma_x = self.lgamma(&x);
            let lgamma_y = self.lgamma(&y);
            let sum = self.add(&x, &y);
            let lgamma_sum = self.lgamma(&sum);
            let t = self.add(&lgamma_x, &lgamma_y);
            self.sub(&t, &lgamma_sum)
        };

        let small_y = self.select(&y_large, &one_large, &both_small);
        self.select(&x_large, &both_large, &small_y)
    }
}
