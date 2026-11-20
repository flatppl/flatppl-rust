data {
  array[6] int y_obs;
}
transformed data {
  int y_obs_n_atom = 0;
  for (i in 1:6) {
    y_obs_n_atom += y_obs[i] == 0;
  }
  array[y_obs_n_atom] int y_obs_atom;
  array[6 - y_obs_n_atom] int y_obs_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:6) {
      if (y_obs[i] == 0) {
        y_obs_atom[a] = i;
        a += 1;
      } else {
        y_obs_rest[b] = i;
        b += 1;
      }
    }
  }
}
parameters {
  real<lower=0> lam;
}
model {
  lam ~ gamma(2.0, 1.0);
  target += y_obs_n_atom * log_mix(0.25, 0, poisson_lpmf(0 | lam));
  target += (6 - y_obs_n_atom) * log(0.75) + poisson_lpmf(y_obs[y_obs_rest] | lam);
}
