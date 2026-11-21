data {
  int<lower=0> N;
  vector[N] X;
  vector[dims(X)[1]] y;
}
parameters {
  real mu;
}
model {
  y ~ normal(mu, 1.0);
}
