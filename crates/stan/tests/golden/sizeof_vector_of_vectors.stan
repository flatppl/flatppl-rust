data {
  int<lower=0> N;
  int<lower=0> N_x_2;
  real y;
  array[N] vector[N_x_2] x;
}
parameters {
  real mu;
}
model {
  y ~ normal(dims(x)[1] + size(x) + sum(linspaced_int_array(N, 1, N)) + sum(dims(x)[1:1]) + mu, 1.0);
}
