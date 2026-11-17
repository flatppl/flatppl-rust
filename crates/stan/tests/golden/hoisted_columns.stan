data {
  int<lower=0> N_Z_2;
  int<lower=0> N;
  vector[N] y_obs;
  array[N] int work;
  matrix[N, N_Z_2] Z;
  int<lower=0> K;
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
  matrix[N, K - 1] W = Z[:, 2:K];
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
