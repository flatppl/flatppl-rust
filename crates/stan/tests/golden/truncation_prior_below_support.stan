data {
  real y;
}
parameters {
  real<lower=0> x;
}
model {
  x ~ exponential(2.0);
  y ~ normal(x, 1);
}
