data {
  vector[3] x_obs;
}
transformed data {
  vector[3] y = x_obs - 1;
}
model {
  if (min(y) < 0) {
    target += negative_infinity();
  }
}
