data {
  int<lower=0> N_k_obs;
  array[N_k_obs] int k_obs;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  for (i in 1:N_k_obs) {
    k_obs[i] ~ binomial(10, p) T[1, ];
  }
}
