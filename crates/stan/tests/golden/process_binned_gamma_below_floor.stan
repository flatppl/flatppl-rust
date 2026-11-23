functions {
  real gamma_interval_lmass(real lo, real hi, real p1, real p2) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= 0) {
      return negative_infinity();
    }
    if (lo <= 0) {
      return gamma_lcdf(hi | p1, p2);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = gamma_lccdf(lo | p1, p2);
    if (la < log(0.5)) {
      return log_diff_exp(la, gamma_lccdf(hi | p1, p2));
    }
    return log_diff_exp(gamma_lcdf(hi | p1, p2), gamma_lcdf(lo | p1, p2));
  }
  real flatppl_binned_poisson_log_lpmf(array[] int n, vector lm) {
    real lp = 0;
    for (k in 1:num_elements(n)) {
      if (lm[k] == negative_infinity()) {
        lp += n[k] == 0 ? 0.0 : negative_infinity();
      } else {
        lp += poisson_log_lpmf(n[k] | lm[k]);
      }
    }
    return lp;
  }
}
data {
  int<lower=0> N;
  array[N] int y;
}
parameters {
  real<lower=0> r;
}
model {
  y ~ flatppl_binned_poisson_log([negative_infinity(), log(10) + gamma_lcdf(1 | 2.0, r), log(10) + gamma_interval_lmass(1, 2, 2.0, r)]');
}
