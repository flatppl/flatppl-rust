data {
  int<lower=0> N;
  int<lower=0> N_X_2;
  matrix[N, N_X_2] X;
  matrix[dims(X)[1], dims(X)[2]] y;
}
parameters {
  real mu;
}
model {
  to_vector(y) ~ normal(mu, 1.0);
}
