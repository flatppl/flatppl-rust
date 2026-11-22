data {
  real y;
}
parameters {
  real ta;
}
model {
  ta ~ std_normal();
  y ~ normal(ta, 1.0);
}
