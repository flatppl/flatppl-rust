data {
  int<lower=0> N_y_1;
  int<lower=0> N_y_2;
  matrix[N_y_1, N_y_2] y;
  int<lower=0> T;
}
transformed data {
  array[T - 1] int steps = linspaced_int_array(T - 1, 1, T - 1);
  matrix[N_y_1, T - 1] change = y[:, 2:T] - y[:, steps];
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
