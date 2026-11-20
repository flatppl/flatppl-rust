data {
  real y;
}
parameters {
  real t;
  real s;
}
model {
  y ~ normal(t + 2.0 * s, 1.0);
}
