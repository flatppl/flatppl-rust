data {
  array[4] int k_obs;
}
parameters {
  real<lower=0> lam;
}
model {
  lam ~ exponential(0.5);
  for (i in 1:4) {
    target += log_sum_exp([log(0.4) + poisson_lpmf(k_obs[i] | lam), log(0.6) + (k_obs[i] > 4 ? negative_infinity() : binomial_lpmf(k_obs[i] | 4, 0.3))]');
  }
}
