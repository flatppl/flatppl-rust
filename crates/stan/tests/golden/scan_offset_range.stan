data {
  int<lower=0> N;
  int<lower=0> N_w;
  real z;
  int<lower=0> T;
  vector[N] y_obs;
  vector[N_w] w;
}
parameters {
  real mu;
}
model {
  vector[T - 1] run;
  {
    real run_state = 0.0;
    for (t in 2:T) {
      run_state = 0.5 * run_state + y_obs[t] - mu;
      run[t - 1] = run_state;
    }
  }
  vector[T - 1] rows_b;
  {
    real a = 0.0;
    for (t in 2:T) {
      rows_b[t - 1] = a - y_obs[t];
      a = a + y_obs[t] * mu;
    }
  }

  mu ~ std_normal();
  z ~ normal(sum(run), 1);
  w ~ normal(rows_b, 1);
}
