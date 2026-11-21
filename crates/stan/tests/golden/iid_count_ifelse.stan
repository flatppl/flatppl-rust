data {
  vector[(1 ? 3 : 2)] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1);
}
