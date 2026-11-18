functions {
  real g(real x, real c) {
    return x > c ? x : c;
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
  for (i in 1:N) {
    for (j in 1:2) {
      m[i, j] = g(x_obs[i], b[j]);
    }
  }

  b ~ std_normal();
  y_obs ~ normal(m[:, 1] - m[:, 2], 1);
}
