data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x_obs;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + x_obs, 1);
}
