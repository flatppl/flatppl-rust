functions {
  real uniform_interval_lmass(real lo, real hi, real p1, real p2) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= p1) {
      return negative_infinity();
    }
    if (lo >= p2) {
      return negative_infinity();
    }
    if (lo <= p1 && hi >= p2) {
      return 0;
    }
    if (lo <= p1) {
      return uniform_lcdf(hi | p1, p2);
    }
    if (hi >= p2) {
      return uniform_lccdf(lo | p1, p2);
    }
    return log_diff_exp(uniform_lcdf(hi | p1, p2), uniform_lcdf(lo | p1, p2));
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
  real<lower=0> s;
  real<lower=0> u;
}
model {
  y ~ flatppl_binned_poisson_log([log(s) + uniform_interval_lmass(0.0, 1, 0.0, u), log(s) + uniform_interval_lmass(1, 2, 0.0, u)]');
}
