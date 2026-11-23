data {
  vector[3] y_obs;
}
parameters {
  real mu;
  real<lower=0> r;
}
model {
  mu ~ std_normal();
  r ~ exponential(1);
  for (i in 1:3) {
    target += log_sum_exp([log(0.4) + normal_lpdf(y_obs[i] | mu, 1.0), log(0.6) + (y_obs[i] < 0 || y_obs[i] > 3.0 ? negative_infinity() : exponential_lpdf(y_obs[i] | r) - exponential_lcdf(3.0 | r))]');
  }
}
