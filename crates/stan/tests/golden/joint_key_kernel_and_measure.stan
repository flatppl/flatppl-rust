data {
  real y;
  real y_2;
}
transformed data {
  real mu = 0.3;
}
model {
  y ~ normal(mu, 1.0);
  y_2 ~ normal(mu, 2.0);
}
