transformed data {
  vector[2] y = [1.27, 0.4]';
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  y ~ normal(mu, sigma);
}
