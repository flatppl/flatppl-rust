functions {
  vector predict(real a, real b, vector x) {
    return a + b * x;
  }
}
data {
  int<lower=0> N;
  array[N] int y_data;
  vector[N] x_data;
}
parameters {
  real a0;
  real b0;
}
model {
  vector[N] eta = predict(a0, b0, x_data);

  a0 ~ std_normal();
  b0 ~ std_normal();
  y_data ~ poisson_log(eta);
}
