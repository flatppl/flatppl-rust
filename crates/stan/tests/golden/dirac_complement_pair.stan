data {
  array[5] int y_obs;
  int<lower=0> K;
}
transformed data {
  int y_obs_n_atom = 0;
  for (i in 1:5) {
    y_obs_n_atom += y_obs[i] == 0;
  }
  array[y_obs_n_atom] int y_obs_atom;
  array[5 - y_obs_n_atom] int y_obs_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:5) {
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
  real<lower=0, upper=1> p;
  real<lower=0, upper=1> psi;
}
model {
  p ~ beta(1.5, 1.5);
  psi ~ beta(1.5, 1.5);
  target += y_obs_n_atom * log_mix(psi, binomial_lpmf(0 | K, p), 0);
  target += (5 - y_obs_n_atom) * log(psi) + binomial_lpmf(y_obs[y_obs_rest] | K, p);
}
