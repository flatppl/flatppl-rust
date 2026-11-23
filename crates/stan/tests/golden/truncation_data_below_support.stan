data {
  vector[2] y_obs;
}
parameters {
  real<lower=0> rate;
}
model {
  rate ~ gamma(2, 1);
  if (min(y_obs) < 0) {
    target += negative_infinity();
  } else {
    y_obs ~ exponential(rate);
  }
}
