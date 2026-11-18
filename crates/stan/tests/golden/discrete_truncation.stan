data {
  array[3] int k_obs;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  k_obs ~ binomial(10, p) T[1, ];
}
