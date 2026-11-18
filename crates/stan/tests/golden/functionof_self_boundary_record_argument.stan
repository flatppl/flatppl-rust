data {
  real y;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + 2 * 1.0, 1);
}
