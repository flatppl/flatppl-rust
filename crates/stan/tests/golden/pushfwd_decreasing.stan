data {
  int k;
}
parameters {
  real<lower=0> lam;
}
model {
  inv(lam) ~ gamma(3, 2);
  target += -2 * log(abs(lam));
  k ~ poisson(lam);
}
