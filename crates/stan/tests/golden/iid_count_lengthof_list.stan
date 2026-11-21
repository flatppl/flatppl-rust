data {
  int<lower=0> N;
  int<lower=0> N_x_2;
  array[N] vector[N_x_2] x;
  vector[size(x)] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1);
}
