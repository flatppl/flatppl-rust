data {
  vector[4] t_obs;
}
parameters {
  real<lower=0> rate;
  real<lower=0> tau;
}
model {
  rate ~ gamma(2.0, 1.0);
  tau ~ gamma(4.0, 1.0);
  if (min(t_obs) < 0.0 || max(t_obs) > tau) {
    target += negative_infinity();
  } else {
    t_obs ~ exponential(rate);
  }
}
