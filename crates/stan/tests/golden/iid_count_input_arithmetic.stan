data {
  int<lower=1> k;
  vector[(k + 1)] y;
}
parameters {
  real mu;
}
model {
  y ~ normal(mu, 1.0);
}
