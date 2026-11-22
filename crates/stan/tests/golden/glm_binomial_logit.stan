data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int y;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  real alpha;
  vector[2] beta;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  y ~ binomial_logit_glm(4, x, alpha, beta);
}
