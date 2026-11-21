data {
  int k;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  logit(p) ~ normal(0.2, 1.1);
  target += -log(p) - log1m(p);
  k ~ binomial(10, p);
}
