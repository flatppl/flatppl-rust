data {
  int<lower=0> N;
  real y;
  vector[N] q;
}
parameters {
  real mu;
}
model {
  y ~ normal(size(rep_array(q, 2)) + mu, 1.0);
}
