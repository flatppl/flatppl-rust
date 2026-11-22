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
  real<lower=0> psi;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  psi ~ exponential(1);
  y ~ neg_binomial_2_log_glm(x, alpha, beta, psi);
}
