data {
  int<lower=0> J;
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_x;
  vector[N_x] x;
}
parameters {
  vector[J] theta;
  real<lower=0> sigma;
}
model {
  theta ~ std_normal();
  sigma ~ exponential(1.0);
  y_obs ~ normal(theta .* x, sigma);
}
