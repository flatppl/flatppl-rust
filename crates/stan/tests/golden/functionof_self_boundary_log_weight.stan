data {
  real y;
}
parameters {
  real mu;
}
model {
  target += -0.5 * mu ^ 2;
  y ~ normal(mu, 1);
}
