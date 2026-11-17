data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] cut;
}
parameters {
  real<lower=0> s;
}
model {
  s ~ exponential(1.0);
  for (i in 1:N) {
    y_obs[i] ~ normal(cut[i] > 1.0 ? cut[i] : 0.0, s);
  }
}
