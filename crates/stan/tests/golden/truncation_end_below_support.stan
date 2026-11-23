data {
  real x;
}
model {
  if (x < 0) {
    target += negative_infinity();
  } else {
    x ~ exponential(2.0);
  }
}
