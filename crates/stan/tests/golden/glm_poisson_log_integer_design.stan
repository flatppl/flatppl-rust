data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int y;
  array[N_x_1, N_x_2] int x;
}
parameters {
  real alpha;
  vector[2] beta;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  y ~ poisson_log_glm(to_matrix(x), alpha, beta);
}
