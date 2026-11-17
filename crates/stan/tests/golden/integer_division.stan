data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_n_success;
  array[N_n_success] int n_success;
  int<lower=0> N_n_trials;
  array[N_n_trials] int n_trials;
  int<lower=0> k;
  int<lower=0> m;
}
transformed data {
  vector[N_n_success] rates = to_vector(n_success) ./ to_vector(n_trials);
  real half = 1.0 * k / m;
  real q = 1.0 * k / m;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y_obs ~ normal(rates + mu + half - q, 1.0);
}
