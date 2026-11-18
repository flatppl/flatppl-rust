data {
  vector[3] y_obs;
}
parameters {
  positive_ordered[2] mu;
}
model {
  mu ~ normal([3, 10]', 1);
  y_obs ~ normal(mu[1], mu[2]);
}
