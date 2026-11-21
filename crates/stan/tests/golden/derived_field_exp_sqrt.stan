data {
  real z;
}
parameters {
  real<lower=1> u;
}
model {
  square(log(u)) ~ inv_gamma(3, 2);
  target += -log(u);
  target += log(2) + log(log(u));
  z ~ normal(u, 1);
}
