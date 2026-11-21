data {
  int<lower=0> N;
  int<lower=0> N_yd_2;
  matrix[N, N_yd_2] yd;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  to_vector(yd) ~ normal(a, 1.0);
}
