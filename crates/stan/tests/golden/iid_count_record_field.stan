data {
  int<lower=0> N;
  vector[N] u;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  u ~ normal(a, 1);
}
