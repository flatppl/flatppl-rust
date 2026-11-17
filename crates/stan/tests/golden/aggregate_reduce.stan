data {
  int<lower=0> K;
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_X_1;
  int<lower=0> N_X_2;
  matrix[N_X_1, N_X_2] X;
  int<lower=0> N_counts_1;
  int<lower=0> N_counts_2;
  array[N_counts_1, N_counts_2] int counts;
}
transformed data {
  vector[N_X_2] means;
  for (i in 1:N_X_2) {
    means[i] = mean(X[:, i]);
  }
  matrix[N_X_1, N_X_2] Xc = X - rep_matrix(means', N_X_1);
  matrix[N_X_1, N_X_2] step;
  for (i in 1:N_X_1) {
    for (j in 1:N_X_2) {
      step[i, j] = X[i, j] >= 1.0 ? 1.0 : 0.0;
    }
  }
  array[N_counts_1] int totals;
  for (i in 1:N_counts_1) {
    totals[i] = sum(counts[i, :]);
  }
}
parameters {
  vector[K] b;
  real<lower=0> sigma;
}
model {
  b ~ std_normal();
  sigma ~ exponential(1.0);
  y_obs ~ normal(Xc * b + step * b + to_vector(totals), sigma);
}
