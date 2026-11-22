data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  vector[N] y;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  real alpha;
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  sigma ~ exponential(1);
  y ~ normal_id_glm(x, alpha, beta, sigma);
}
