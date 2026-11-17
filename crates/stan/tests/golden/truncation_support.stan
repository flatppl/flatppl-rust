data {
  int<lower=0> N_x_obs;
  vector<lower=-1.0, upper=2.0>[N_x_obs] x_obs;
}
parameters {
  real<lower=0> s;
}
model {
  x_obs ~ normal(0, s);
}
