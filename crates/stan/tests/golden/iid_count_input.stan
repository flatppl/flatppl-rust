data {
  int<lower=1> n;
  vector[n] b;
}
parameters {
  real mu;
}
model {
  b ~ normal(mu, 1.0);
}
