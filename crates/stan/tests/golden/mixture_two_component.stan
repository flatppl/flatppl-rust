data {
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  real<lower=0, upper=1> theta;
  vector[2] mu;
}
model {
  theta ~ beta(2, 2);
  mu ~ normal(0, 5);
  for (i in 1:N) {
    target += log_mix(theta, normal_lpdf(y_obs[i] | mu[1], 1), normal_lpdf(y_obs[i] | mu[2], 1));
  }
}
