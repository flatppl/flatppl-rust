data {
  real y;
}
transformed data {
  real mu = 0.3;
}
model {
  y ~ normal(mu, 1.0);
}
