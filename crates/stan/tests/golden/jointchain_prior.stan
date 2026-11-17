data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y_obs;
  array[N] int group;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> tau;
  real mu;
  vector[J] theta;
}
model {
  theta ~ normal(mu, tau);
  y_obs ~ normal(theta[group], sigma);
}
