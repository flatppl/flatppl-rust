data {
  vector[3] x_obs;
}
model {
  if (min(x_obs) < 0) {
    target += negative_infinity();
  }
}
