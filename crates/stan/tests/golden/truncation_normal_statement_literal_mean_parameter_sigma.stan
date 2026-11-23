data {
  real y;
}
parameters {
  real<lower=0> s;
}
model {
  y ~ normal(1.0, s) T[0.0, ];
}
