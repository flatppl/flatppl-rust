data {
  int<lower=0> len;
  vector[len] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1.0);
}
