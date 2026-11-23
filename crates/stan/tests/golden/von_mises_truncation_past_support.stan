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
    y ~ von_mises(mu, 2.0);
    target += -von_mises_lccdf(-1.0 | mu, 2.0);
  }
}
