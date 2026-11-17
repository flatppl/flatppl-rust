data {
  int<lower=0> N;
  int<lower=0> N_y_obs;
  vector[N] z;
  vector[N_y_obs] y_obs;
  vector[N_y_obs] x;
}
parameters {
  real mu;
  real beta;
}
model {
  mu ~ normal(mean(z), 1.0);
  beta ~ std_normal();
  y_obs ~ normal(mu + beta * x, 1.0);
}
