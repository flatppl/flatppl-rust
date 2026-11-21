data {
  int<lower=0> N;
  real y;
  vector[N] X;
}
parameters {
  real mu;
}
model {
  y ~ normal(sum(dims(X)[1:1]) + dims(X)[1] + sum(dims(X')[2:2]) + mu, 1.0);
}
