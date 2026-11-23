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
  y ~ flatppl_binned_poisson_log([log(s) + uniform_lcdf(2 | 0.0, 3.0), log(s) + uniform_lccdf(2 | 0.0, 3.0)]');
}
