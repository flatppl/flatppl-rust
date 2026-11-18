data {
  array[5] int y_obs;
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
  real<lower=0, upper=1> hu;
  real<lower=0> mu;
}
model {
  hu ~ beta(2, 2);
  mu ~ gamma(2, 1);
  target += y_obs_n_atom * log(hu);
  target += (5 - y_obs_n_atom) * log1m(hu) + (5 - y_obs_n_atom > 0 && (min(y_obs[y_obs_rest]) < 1) ? negative_infinity() : poisson_lpmf(y_obs[y_obs_rest] | mu) - (5 - y_obs_n_atom) * poisson_lccdf(0 | mu));
}
