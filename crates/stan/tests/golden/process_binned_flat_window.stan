functions {
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
}
model {
  y ~ flatppl_binned_poisson_log([log_sum_exp(log(s) + uniform_lcdf(1 | 0.0, 6.0), log(0.3) + log(0.5)), log_sum_exp(log(s) + log_diff_exp(uniform_lcdf(3 | 0.0, 6.0), uniform_lcdf(1 | 0.0, 6.0)), log(0.3) + log(2)), log_sum_exp(log(s) + uniform_lccdf(3 | 0.0, 6.0), log(0.3) + log(2))]');
}
