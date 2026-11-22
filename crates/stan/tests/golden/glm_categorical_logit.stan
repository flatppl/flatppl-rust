data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int y;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  vector[3] alpha;
  matrix[2, 3] beta;
}
model {
  alpha ~ std_normal();
  to_vector(beta) ~ std_normal();
  y ~ categorical_logit_glm(x, alpha, beta);
}
