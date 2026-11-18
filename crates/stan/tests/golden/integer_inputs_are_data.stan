data {
  int<lower=0, upper=1> active;
  array[3] int v;
  real y;
}
parameters {
  real mu;
}
model {
  real center = active ? 2.0 : -3.0;

  y ~ normal(center + sum(v) + mu, 2.0);
}
