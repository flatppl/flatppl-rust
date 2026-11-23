functions {
  real flatppl_lphi(real z) {
    // log Phi(z) from erfc for z >= -37: Stan's normal_lcdf derivative is an approximation.
    if (z > 0) {
      return log1m(0.5 * erfc(z / sqrt2()));
    }
    if (z >= -37) {
      return log(erfc(-z / sqrt2())) - log2();
    }
    return std_normal_lcdf(z);
  }
  real exponential_interval_lmass(real lo, real hi, real p1) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= 0) {
      return negative_infinity();
    }
    if (lo <= 0) {
      return exponential_lcdf(hi | p1);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = exponential_lccdf(lo | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, exponential_lccdf(hi | p1));
    }
    return log_diff_exp(exponential_lcdf(hi | p1), exponential_lcdf(lo | p1));
  }
}
data {
  real x;
  real s;
}
model {
  target += log_sum_exp([log(0.4) + (x < 0.0 ? negative_infinity() : normal_lpdf(x | 0.0, s)), log(0.6) + (x < 0.5 || x > 3.0 ? negative_infinity() : exponential_lpdf(x | 2.0))]') - log_sum_exp([log(0.4) + flatppl_lphi(0 / s), log(0.6) + exponential_interval_lmass(0.5, 3.0, 2.0)]');
}
