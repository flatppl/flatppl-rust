data {
  vector[2] y_obs;
}
parameters {
  real<lower=0> rate;
}
model {
  rate ~ gamma(2.0, 1.0);
  if (min(y_obs) < 0 || max(y_obs) > 2.0) {
    target += negative_infinity();
  } else {
    y_obs ~ exponential(rate);
  }
}
