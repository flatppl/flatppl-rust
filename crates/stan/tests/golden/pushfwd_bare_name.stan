data {
  real z;
}
parameters {
  real v;
  real<lower=0> w;
}
model {
  exp(v) ~ exponential(1.5);
  target += v;
  square(w) ~ inv_gamma(5, 5);
  target += log(2) + log(w);
  z ~ normal(v, w);
}
