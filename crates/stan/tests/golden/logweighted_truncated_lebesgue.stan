data {
  int<lower=0> n;
  vector[n] y_obs;
}
transformed data {
  real lo = mean(y_obs) - 3 * sd(y_obs);
  real hi = mean(y_obs) + 3 * sd(y_obs);
}
parameters {
  vector<lower=lo, upper=hi>[n] mu;
  positive_ordered[2] sigma;
}
model {
  target += normal_lpdf(mu | 7.0, 1.0) + student_t_lpdf(sigma | 4, 0, 1);
  y_obs ~ normal(mu, sigma[2]);
}
