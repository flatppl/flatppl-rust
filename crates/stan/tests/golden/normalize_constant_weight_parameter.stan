data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
}
parameters {
  real mu;
  real<lower=0> sigma;
  real<lower=0> nu;
}
model {
  mu ~ std_normal();
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);
  y_obs ~ student_t(nu, mu + x, sigma);
}
