data {
  real obs_a;
  real obs_b;
}
parameters {
  real mu;
}
model {
  mu ~ normal(0.0, 2.0);
  obs_a ~ normal(mu, 1.0);
  obs_b ~ normal(2.0 * mu, 0.5);
}
