data {
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  real mu;
  real phi;
  real theta;
}
model {
  vector[N] states_mean;
  {
    real previous = mu;
    real error = 0.0;
    for (t in 1:N) {
      states_mean[t] = mu + phi * previous + theta * error;
      error = y_obs[t] - (mu + phi * previous + theta * error);
      previous = y_obs[t];
    }
  }

  mu ~ std_normal();
  phi ~ std_normal();
  theta ~ std_normal();
  y_obs ~ normal(states_mean, 1);
}
