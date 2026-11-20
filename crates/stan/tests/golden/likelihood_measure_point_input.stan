data {
  real y;
}
parameters {
  real t;
}
model {
  y ~ normal(t, 1.0);
}
