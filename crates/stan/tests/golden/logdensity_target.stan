data {
  int<lower=0> N_x;
  vector[N_x] x;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  x ~ normal(mu, sigma);
}
