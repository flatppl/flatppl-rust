data {
  vector[3] x;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  x ~ normal(mu, sigma);
}
