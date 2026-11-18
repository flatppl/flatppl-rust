data {
  int<lower=0> N;
  array[N] int counts;
  vector[2] y_obs;
}
parameters {
  real<lower=0> rate;
}
model {
  target += poisson_lpmf(counts | rate) + poisson_lpmf(3 | rate);
  y_obs ~ normal(rate, 1.0);
}
