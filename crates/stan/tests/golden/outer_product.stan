data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N;
  int<lower=0> N_x;
  vector[N_x] x;
}
transformed data {
  matrix[N_x, N_x] distance_sq;
  for (i in 1:N_x) {
    for (j in 1:N_x) {
      distance_sq[i, j] = (x[i] - x[j]) ^ 2;
    }
  }
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
}
model {
  matrix[N_x, N_x] covariance = alpha ^ 2 * exp(-0.5 * distance_sq / rho ^ 2) + 0.01 * identity_matrix(N);

  rho ~ gamma(25, 4);
  alpha ~ normal(0, 2) T[0, ];
  y_obs ~ multi_normal(zeros_vector(N), covariance);
}
