data {
  array[2] vector[2] x_obs;
}
parameters {
  real mu;
}
model {
  {
    int outside = 0;
    for (i in 1:2) {
      outside = outside || min(x_obs[i]) < 0;
    }
    if (outside) {
      target += negative_infinity();
    } else {
      for (i in 1:2) {
        x_obs[i] ~ normal(mu, 1);
      }
    }
  }
}
