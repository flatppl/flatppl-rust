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
  real<lower=0> rate;
}
model {
  real lmass = gamma_lccdf(1000 | 1.0, rate);

  y ~ flatppl_binned_poisson_log([gamma_lccdf(1001 | 1.0, rate) - lmass]');
}
