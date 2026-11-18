data {
  array[4] int k_obs;
}
parameters {
  real<lower=0> lam;
}
model {
  lam ~ gamma(2, 1);
  k_obs ~ poisson(lam) T[1, ];
}
