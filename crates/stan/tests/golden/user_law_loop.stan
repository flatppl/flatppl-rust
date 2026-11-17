data {
  vector[4] y_obs;
  vector[4] cut;
}
parameters {
  real<lower=0> s;
}
model {
  s ~ exponential(1.0);
  for (i in 1:4) {
    y_obs[i] ~ normal(cut[i] > 1.0 ? cut[i] : 0.0, s);
  }
}
