data {
  int<lower=0> N;
  int<lower=0> N_X_2;
  array[N] vector[N_X_2] X;
  vector[dims(X)[1]] y;
}
parameters {
  real mu;
}
model {
  y ~ normal(mu, 1.0);
}
