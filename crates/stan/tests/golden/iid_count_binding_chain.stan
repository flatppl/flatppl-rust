data {
  int<lower=0> N;
  vector[N] yd;
}
transformed data {
  int n1 = num_elements(yd);
  int n3 = n1 + 0;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  yd ~ normal(a, 1);
}
