data {
  real y;
}
parameters {
  real<lower=0> r;
}
model {
  if (y < 0.0 || y > 10.0) {
    target += negative_infinity();
  } else {
    y ~ exponential(r);
    target += -exponential_lcdf(10.0 | r);
  }
}
