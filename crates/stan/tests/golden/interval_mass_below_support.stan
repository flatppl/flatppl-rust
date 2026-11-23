data {
  real y;
}
model {
  if (y < 0 || y > 2.0) {
    target += negative_infinity();
  } else {
    y ~ exponential(1.0);
    target += -exponential_lcdf(2.0 | 1.0);
  }
}
