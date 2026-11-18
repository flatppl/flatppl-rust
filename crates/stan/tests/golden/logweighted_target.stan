data {
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 2);
  sigma ~ exponential(1.0);
  target += normal_lpdf(y_obs | mu, sigma);
}
