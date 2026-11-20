parameters {
  real mu;
  real x;
}
model {
  if ((x - 1) / 2 < -pi() || (x - 1) / 2 > pi()) {
    target += negative_infinity();
  } else {
    (x - 1) / 2 ~ von_mises(mu, 2.0);
    target += -log(2);
  }
}
