functions {
  real exponential_interval_lmass(real lo, real hi, real p1) {
    real a = fmax(lo, 0);
    real b = fmax(hi, 0);
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = exponential_lccdf(a | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, exponential_lccdf(b | p1));
    }
    return log_diff_exp(exponential_lcdf(b | p1), exponential_lcdf(a | p1));
  }
}
data {
  real x;
  real s;
}
model {
  target += log_sum_exp([log(0.4) + (x < 0.0 ? negative_infinity() : normal_lpdf(x | 0.0, s)), log(0.6) + (x < 0.5 || x > 3.0 ? negative_infinity() : exponential_lpdf(x | 2.0))]') - log_sum_exp([log(0.4) + normal_lccdf(0.0 | 0.0, s), log(0.6) + exponential_interval_lmass(0.5, 3.0, 2.0)]');
}
