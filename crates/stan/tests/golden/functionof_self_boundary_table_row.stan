data {
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  real<lower=0, upper=1> q;
}
model {
  vector[N] states_current;
  {
    real next = 0.5;
    for (t in 1:N) {
      states_current[t] = next;
      next = q * next + (1 - q) * y_obs[t];
    }
  }
  vector[N] values;
  for (i in 1:N) {
    values[i] = 2 * states_current[i];
  }

  q ~ beta(2, 2);
  y_obs ~ normal(values, 1);
}
