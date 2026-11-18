data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x_obs;
}
transformed data {
  vector[N] run;
  {
    real run_state = 0.0;
    for (t in 1:N) {
      run_state = 0.5 * run_state + x_obs[t];
      run[t] = run_state;
    }
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + run, 1);
}
