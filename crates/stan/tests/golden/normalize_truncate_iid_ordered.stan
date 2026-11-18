data {
  int<lower=0> K;
  vector[3] y_obs;
}
parameters {
  ordered[K] mu;
}
model {
  mu ~ normal(0, 2);
  target += lgamma(K + 1);
  y_obs ~ normal(mu[3] - mu[1], 1);
}
