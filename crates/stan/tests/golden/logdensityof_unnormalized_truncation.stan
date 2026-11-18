data {
  real y;
}
parameters {
  real mu;
}
model {
  target += (mu < 0 ? negative_infinity() : std_normal_lpdf(mu));
  y ~ normal(mu, 1);
}
