data {
  array[2] vector<lower=0>[2] x_obs;
}
parameters {
  real mu;
}
model {
  for (i in 1:2) {
    x_obs[i] ~ normal(mu, 1);
  }
}
