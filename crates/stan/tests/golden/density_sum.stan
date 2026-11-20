data {
  int k;
  real y;
}
parameters {
  real a;
  real<lower=0> b;
  real x;
}
model {
  x ~ normal(a, b);
  k ~ poisson(b);
  y ~ exponential(b);
}
