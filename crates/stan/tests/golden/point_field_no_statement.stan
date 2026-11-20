parameters {
  real x;
}
model {
  if (x < 0.0 || x > 1.0) {
    target += negative_infinity();
  }
  target += -0.5 * x ^ 2;
}
