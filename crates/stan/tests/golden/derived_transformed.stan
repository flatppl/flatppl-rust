data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_x;
  vector[N_x] x;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> tau;
}
transformed parameters {
  real sigma = 1 / sqrt(tau);
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  tau ~ gamma(2.0, 1.0);
  y_obs ~ normal(alpha + beta * x, sigma);
}
