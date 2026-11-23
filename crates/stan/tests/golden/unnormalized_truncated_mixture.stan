functions {
  real poisson_interval_lmass(int lo, int hi, real p1) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo > hi) {
      return negative_infinity();
    }
    if (hi < 0) {
      return negative_infinity();
    }
    if (lo <= 0) {
      return poisson_lcdf(hi | p1);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = poisson_lccdf(lo - 1 | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, poisson_lccdf(hi | p1));
    }
    return log_diff_exp(poisson_lcdf(hi | p1), poisson_lcdf(lo - 1 | p1));
  }
}
data {
  int<lower=0> k;
}
model {
  target += log_sum_exp([log(0.3) + (k < 1 || k > 4 ? negative_infinity() : poisson_lpmf(k | 3.0)), log(0.7) + (k < 0 || k > 2 ? negative_infinity() : poisson_lpmf(k | 1.5))]') - log_sum_exp([log(0.3) + poisson_interval_lmass(1, 4, 3.0), log(0.7) + poisson_lcdf(2 | 1.5)]');
}
