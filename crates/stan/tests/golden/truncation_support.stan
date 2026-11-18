data {
  vector[2] x_obs;
}
parameters {
  real<lower=0> s;
}
model {
  if (min(x_obs) < -1.0 || max(x_obs) > 2.0) {
    target += negative_infinity();
  } else {
    x_obs ~ normal(0, s);
  }
}
