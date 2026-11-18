data {
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  real<lower=0, upper=1> q;
  ordered[2] mu;
}
model {
  vector[N] states_current;
  {
    real next = 0.5;
    for (t in 1:N) {
      states_current[t] = next;
      next = q + (1 - 2 * q) * inv_logit(logit(next) + y_obs[t]);
    }
  }

  q ~ beta(2, 2);
  mu ~ normal([0, 3]', 1);
  for (i in 1:N) {
    target += log_mix(states_current[i], normal_lpdf(y_obs[i] | mu[1], 1), normal_lpdf(y_obs[i] | mu[2], 1));
  }
}
