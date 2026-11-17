data {
  int<lower=0> N_k_obs;
  array[N_k_obs] int k_obs;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  p ~ beta(2, 2);
  k_obs ~ neg_binomial(1, p / (1 - p));
}
