data {
  int<lower=0> N;
  int<lower=0> N_X_2;
  matrix[N, N_X_2] X;
  matrix[dims(X')[1], dims(X')[2]] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  to_vector(b) ~ normal(a + sum(X), 1.0);
}
