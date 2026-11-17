data {
  int<lower=0> N;
  int<lower=0> J;
  matrix[J, N] y;
  int<lower=0> T;
}
transformed data {
  array[T - 1] int steps = linspaced_int_array(T - 1, 1, T - 1);
  matrix[J, T - 1] change = y[:, 2:T] - y[:, steps];
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ std_normal();
  sigma ~ exponential(1.0);
  to_vector(change) ~ normal(mu, sigma);
}
