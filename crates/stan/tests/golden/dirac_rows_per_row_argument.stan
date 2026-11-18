data {
  int<lower=0> N;
  int<lower=0> T;
  int<lower=0> M;
  array[N, T] int y_obs;
}
transformed data {
  int y_obs_n_atom = 0;
  for (i in 1:N) {
    y_obs_n_atom += max(y_obs[i]) == 0 && min(y_obs[i]) == 0;
  }
  array[y_obs_n_atom] int y_obs_atom;
  array[N - y_obs_n_atom] int y_obs_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:N) {
      if (max(y_obs[i]) == 0 && min(y_obs[i]) == 0) {
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
  real<lower=0, upper=1> omega;
  vector<lower=0, upper=1>[T] mean_p;
  real<lower=0, upper=5> sigma;
  vector[M] eps_raw;
}
model {
  matrix[M, T] lp = rep_matrix(sigma * eps_raw, T) + rep_matrix(logit(mean_p)', M);

  omega ~ uniform(0, 1);
  mean_p ~ uniform(0, 1);
  sigma ~ uniform(0, 5);
  eps_raw ~ std_normal();
  for (i in 1:y_obs_n_atom) {
    target += log_mix(omega, bernoulli_logit_lpmf(y_obs[y_obs_atom[i]] | lp[y_obs_atom[i]]), 0);
  }
  target += (N - y_obs_n_atom) * log(omega) + bernoulli_logit_lpmf(to_array_1d(y_obs[y_obs_rest]) | to_vector(lp[y_obs_rest]'));
}
