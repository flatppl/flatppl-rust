data {
  int<lower=0> N;
  array[N] int y_obs;
  vector[N] x;
}
transformed data {
  int y_obs_n_atom = 0;
  for (i in 1:N) {
    y_obs_n_atom += y_obs[i] == 0;
  }
  array[y_obs_n_atom] int y_obs_atom;
  array[N - y_obs_n_atom] int y_obs_rest;
  {
    int a = 1;
    int b_2 = 1;
    for (i in 1:N) {
      if (y_obs[i] == 0) {
        y_obs_atom[a] = i;
        a += 1;
      } else {
        y_obs_rest[b_2] = i;
        b_2 += 1;
      }
    }
  }
}
parameters {
  real<lower=0, upper=1> hu;
  real b;
}
model {
  vector[N] mu = exp(b * x);

  hu ~ beta(2, 2);
  b ~ std_normal();
  target += y_obs_n_atom * log(hu);
  target += (N - y_obs_n_atom) * log1m(hu) + (N - y_obs_n_atom > 0 && (min(y_obs[y_obs_rest]) < 1) ? negative_infinity() : poisson_lpmf(y_obs[y_obs_rest] | mu[y_obs_rest]) - poisson_lccdf(0 | mu[y_obs_rest]));
}
