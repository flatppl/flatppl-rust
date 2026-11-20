data {
  vector[3] y_obs;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  if (min(y_obs) < -pi() || max(y_obs) > pi()) {
    target += negative_infinity();
  } else {
    y_obs ~ von_mises(mu, 2.0);
  }
}
