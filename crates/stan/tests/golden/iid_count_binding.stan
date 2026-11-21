data {
  int<lower=0> n;
  vector[n] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1);
}
