data {
  int<lower=0> N;
  vector[N] x;
  vector[num_elements(x) + 1] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1);
}
