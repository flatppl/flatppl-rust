data {
  int<lower=0> N;
  int<lower=0> N_Y_obs_2;
  matrix[N, N_Y_obs_2] Y_obs;
}
transformed data {
  int r = dims(Y_obs)[1];
  int c = dims(Y_obs)[2];
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  to_vector(Y_obs) ~ normal(mu, 1.0);
}
