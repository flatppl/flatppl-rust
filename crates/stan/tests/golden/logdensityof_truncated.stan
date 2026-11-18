data {
  real v;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  target += (v < 0.0 ? negative_infinity() : normal_lpdf(v | a, 1) - normal_lccdf(0.0 | a, 1));
}
