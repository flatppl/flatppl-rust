data {
  real y;
}
parameters {
  real<lower=0> s;
}
model {
  y ~ normal(0.0, s) T[0.0, ];
}
