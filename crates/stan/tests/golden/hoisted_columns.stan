data {
  int<lower=0> N;
  vector[N] y_obs;
  array[N] int work;
  int<lower=0> K;
  matrix[N, K] Z;
}
transformed data {
  vector[N] X_ifelse;
  for (i in 1:N) {
    X_ifelse[i] = work[i] == 2 ? 1.0 : 0.0;
  }
  vector[N] X_ifelse_2;
  for (i in 1:N) {
    X_ifelse_2[i] = work[i] == 3 ? 1.0 : 0.0;
  }
  matrix[N, 3] X = append_col(append_col(ones_vector(N), X_ifelse), X_ifelse_2);
  array[3] int W_idx;
  for (i in 1:3) {
    W_idx[i] = i + 1;
  }
  matrix[N, 3] W = Z[:, W_idx];
}
parameters {
  vector[3] beta;
  vector[3] gamma;
  real<lower=0> sigma;
}
model {
  beta ~ std_normal();
  gamma ~ std_normal();
  sigma ~ exponential(1.0);
  y_obs ~ normal(X * beta + W * gamma, sigma);
}
