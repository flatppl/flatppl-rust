data {
  real y;
}
parameters {
  real alpha;
  real t;
}
model {
  t ~ normal(alpha, 1.0);
  y ~ normal(t, 1.0);
}
