data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_cut;
  vector[N_cut] cut;
}
parameters {
  real<lower=0> s;
}
model {
  s ~ exponential(1.0);
  for (i in 1:N_y_obs) {
    y_obs[i] ~ normal(cut[i] > 1.0 ? cut[i] : 0.0, s);
  }
}
