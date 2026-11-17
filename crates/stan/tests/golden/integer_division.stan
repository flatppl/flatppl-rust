data {
  vector[3] y_obs;
  array[3] int n_success;
  array[3] int n_trials;
  int<lower=0> k;
  int<lower=0> m;
}
transformed data {
  vector[3] rates = to_vector(n_success) ./ to_vector(n_trials);
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
