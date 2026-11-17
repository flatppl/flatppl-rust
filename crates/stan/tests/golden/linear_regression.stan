data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
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
