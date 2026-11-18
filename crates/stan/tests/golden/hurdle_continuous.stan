data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
}
transformed data {
  int y_obs_n_atom = 0;
  for (i in 1:N) {
    y_obs_n_atom += y_obs[i] == 0.0;
  }
  array[N - y_obs_n_atom] int y_obs_rest;
  {
    int b_2 = 1;
    for (i in 1:N) {
      if (y_obs[i] != 0.0) {
        y_obs_rest[b_2] = i;
        b_2 += 1;
      }
    }
  }
}
parameters {
  real<lower=0, upper=1> hu;
  real<lower=0> shape;
  real b;
}
model {
  vector[N] mu = exp(b * x);

  hu ~ beta(2, 2);
  shape ~ gamma(2, 1);
  b ~ std_normal();
  target += y_obs_n_atom * log(hu);
  if (N - y_obs_n_atom > 0 && (min(y_obs[y_obs_rest]) < 0)) {
    target += negative_infinity();
  } else {
    target += (N - y_obs_n_atom) * log1m(hu) + gamma_lpdf(y_obs[y_obs_rest] | shape, shape / mu[y_obs_rest]);
  }
}
