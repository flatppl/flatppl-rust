data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  int<lower=0> N_cutpoints;
  array[N] int y;
  matrix[N_x_1, N_x_2] x;
  vector[N_cutpoints] cutpoints;
}
parameters {
  vector[2] beta;
}
model {
  beta ~ std_normal();
  y ~ ordered_logistic_glm(x, beta, cutpoints);
}
