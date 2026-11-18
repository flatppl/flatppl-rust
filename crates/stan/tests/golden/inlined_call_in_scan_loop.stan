data {
  int<lower=0> N;
  int<lower=0> N_vs;
  vector[N] y;
  vector[N] xs;
  vector[N_vs] vs;
}
transformed data {
  vector[N] means;
  for (i in 1:N) {
    means[i] = xs[i] * sum(vs * 2.0);
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + means, 1);
}
