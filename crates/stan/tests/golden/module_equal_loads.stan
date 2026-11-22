data {
  real y;
}
parameters {
  real t;
}
model {
  t ~ std_normal();
  y ~ normal(t, 1.0);
}
