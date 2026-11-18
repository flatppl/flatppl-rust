data {
  vector[2] y;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(0.5 * [3.0, 4.0]' + mu, 1);
}
