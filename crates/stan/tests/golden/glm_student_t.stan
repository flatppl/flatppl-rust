data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  vector[N] y;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  real<lower=0> nu;
  real alpha;
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  nu ~ gamma(2.0, 0.5);
  alpha ~ std_normal();
  beta ~ std_normal();
  sigma ~ exponential(1);
  y ~ student_t(nu, alpha + x * beta, sigma);
}
