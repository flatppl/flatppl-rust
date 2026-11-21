data {
  int<lower=0> N;
  vector[N] yd;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  yd ~ normal(a, 1.0);
}
