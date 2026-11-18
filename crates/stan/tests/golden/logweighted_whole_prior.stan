data {
  int<lower=0> M;
  vector[M] y_obs;
}
parameters {
  real<lower=0> kappa;
  vector<lower=0>[M] mu;
}
model {
  kappa ~ normal(0, 0.5) T[0, ];
  target += normal_lpdf(mu | 3.28, kappa);
  y_obs ~ normal(mu, 1.0);
}
