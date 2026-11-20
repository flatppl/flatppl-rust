data {
  real y;
  real y_2;
}
parameters {
  real t;
  real<lower=0> s;
}
model {
  y ~ normal(t, 1.0);
  y_2 ~ normal(0.0, s);
}
