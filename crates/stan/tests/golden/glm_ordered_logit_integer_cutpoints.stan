data {
  int<lower=0> N;
  int<lower=0> N_cutpoints;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int y;
  array[N_cutpoints] int cutpoints;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  vector[2] beta;
}
model {
  beta ~ std_normal();
  y ~ ordered_logistic_glm(x, beta, to_vector(cutpoints));
}
