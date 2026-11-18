functions {
  real poisson_interval_lmass(int lo, int hi, real p1) {
    int a = lo - 1;
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = poisson_lccdf(a | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, poisson_lccdf(hi | p1));
    }
    return log_diff_exp(poisson_lcdf(hi | p1), poisson_lcdf(a | p1));
  }
}
data {
  int<lower=0> k;
}
model {
  target += log_sum_exp([log(0.3) + (k < 1 || k > 4 ? negative_infinity() : poisson_lpmf(k | 3.0)), log(0.7) + (k < 0 || k > 2 ? negative_infinity() : poisson_lpmf(k | 1.5))]') - log_sum_exp([log(0.3) + poisson_interval_lmass(1, 4, 3.0), log(0.7) + poisson_interval_lmass(0, 2, 1.5)]');
}
