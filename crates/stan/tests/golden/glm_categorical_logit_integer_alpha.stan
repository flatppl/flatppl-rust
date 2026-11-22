data {
  int<lower=0> N;
  int<lower=0> N_alpha;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int y;
  array[N_alpha] int alpha;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  matrix[2, 3] beta;
}
model {
  to_vector(beta) ~ std_normal();
  y ~ categorical_logit_glm(x, to_vector(alpha), beta);
}
