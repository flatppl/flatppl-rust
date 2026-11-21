data {
  vector[4] y_obs;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  for (i in 1:4) {
    target += log_sum_exp([log(0.3) + normal_lpdf(y_obs[i] | mu, 2.0), log(0.3) + (y_obs[i] < 0 ? negative_infinity() : gamma_lpdf(y_obs[i] | 2.0, 1.5)), log(0.4) + (y_obs[i] < 0 || y_obs[i] > 1 ? negative_infinity() : beta_lpdf(y_obs[i] | 2.0, 3.0))]');
  }
}
