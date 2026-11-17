data {
  int<lower=0> J;
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_group;
  array[N_group] int group;
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
