data {
  real y;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  if (y < -1.0 || y > pi()) {
    target += negative_infinity();
  } else {
    y ~ von_mises(mu, 2.0) T[-1.0, pi()];
  }
}
