data {
  array[2, 3] int y;
}
parameters {
  real<lower=0> lam;
}
model {
  lam ~ exponential(1);
  to_array_1d(y) ~ poisson(lam);
}
