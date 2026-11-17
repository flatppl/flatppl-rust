data {
  array[3] int k_obs;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  for (i in 1:3) {
    k_obs[i] ~ binomial(10, p) T[1, ];
  }
}
