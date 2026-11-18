data {
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  real mu;
}
model {
  mu ~ normal(0, 5);
  for (i in 1:N) {
    target += log_sum_exp([normal_lpdf(y_obs[i] | mu, 1), normal_lpdf(y_obs[i] | mu + 3, 1)]') - log(2);
  }
}
