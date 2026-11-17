data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
}
transformed data {
  matrix[N, N] distance_sq;
  for (i in 1:N) {
    for (j in 1:N) {
      distance_sq[i, j] = (x[i] - x[j]) ^ 2;
    }
  }
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
}
model {
  matrix[N, N] covariance = alpha ^ 2 * exp(-0.5 * distance_sq / rho ^ 2) + 0.01 * identity_matrix(N);

  rho ~ gamma(25, 4);
  alpha ~ normal(0, 2) T[0, ];
  y_obs ~ multi_normal(rep_vector(0.0, N), covariance);
}
