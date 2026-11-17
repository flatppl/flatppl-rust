data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_x;
  vector[N_x] x;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ std_normal() T[0, ];
  y_obs ~ normal(alpha + beta * x, sigma);
}
