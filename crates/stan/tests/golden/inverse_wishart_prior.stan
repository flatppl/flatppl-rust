data {
  int<lower=0> N;
  int<lower=0> N_S_2;
  int<lower=0> N_y;
  matrix[N, N_S_2] S;
  vector[N_y] y;
}
parameters {
  cov_matrix[3] W;
}
model {
  W ~ inv_wishart(4.0, S);
  y ~ multi_normal([0.0, 0.0, 0.0]', W);
}
