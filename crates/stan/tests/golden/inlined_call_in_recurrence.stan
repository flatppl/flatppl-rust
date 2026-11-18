data {
  int<lower=0> N;
  int<lower=0> N_vs;
  vector[N] y;
  vector[N] xs;
  vector[N_vs] vs;
}
transformed data {
  vector[N] means;
  {
    real means_state = 0.1;
    for (t in 1:N) {
      vector[N_vs] h = vs * means_state;
      means_state = 0.5 * means_state + xs[t] * sum(h) / 10;
      means[t] = means_state;
    }
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + means, 1);
}
