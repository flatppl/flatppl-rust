parameters {
  real mu;
  real xobs;
}
model {
  if (xobs < -pi() || xobs > pi()) {
    target += negative_infinity();
  } else {
    xobs ~ von_mises(mu, 2.0);
  }
}
