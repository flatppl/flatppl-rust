functions {
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
  y ~ flatppl_binned_poisson_log([log(10) + exponential_lcdf(1 | r), log(10) + exponential_interval_lmass(1, 2, r)]');
}
