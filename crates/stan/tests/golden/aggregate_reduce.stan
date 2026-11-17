data {
  int<lower=0> N;
  int<lower=0> N_counts_2;
  int<lower=0> K;
  vector[N] y_obs;
  matrix[N, K] X;
  array[N, N_counts_2] int counts;
}
transformed data {
  vector[K] means;
  for (i in 1:K) {
    means[i] = mean(X[:, i]);
  }
  matrix[N, K] Xc = X - rep_matrix(means', N);
  matrix[N, K] step;
  for (i in 1:N) {
    for (j in 1:K) {
      step[i, j] = X[i, j] >= 1.0 ? 1.0 : 0.0;
    }
  }
  array[N] int totals;
  for (i in 1:N) {
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
