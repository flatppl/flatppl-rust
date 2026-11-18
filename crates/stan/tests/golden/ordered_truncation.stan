data {
  vector[3] y_obs;
}
parameters {
  ordered[2] mu;
}
model {
  mu ~ normal([0, 3]', 1);
  y_obs ~ normal(mu[2] - mu[1], 1);
}
