data {
  real z;
}
parameters {
  real<lower=0> sigma;
}
model {
  square(sigma) ~ inv_gamma(5, 5);
  target += log(2) + log(sigma);
  z ~ normal(0.3, sigma);
}
