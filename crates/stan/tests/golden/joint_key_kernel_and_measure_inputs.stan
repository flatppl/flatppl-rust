data {
  real y;
  real y_2;
}
parameters {
  real t;
}
model {
  y ~ normal(t, 1.0);
  y_2 ~ normal(t, 2.0);
}
