data {
  vector[3] y_data;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0.0, 5.0);
  sigma ~ gamma((2.0 ^ 2 / 1.0 ^ 2), (2.0 / 1.0 ^ 2));
  y_data ~ normal(mu, sigma);
}
