functions {
  vector g(vector x, real c) {
    return c * x + 1;
  }
}
data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x_obs;
}
parameters {
  vector[2] b;
}
model {
  matrix[N, 2] m;
  for (i in 1:2) {
    m[:, i] = g(x_obs, b[i]);
  }

  b ~ std_normal();
  y_obs ~ normal(m[:, 1] - m[:, 2], 1);
}
