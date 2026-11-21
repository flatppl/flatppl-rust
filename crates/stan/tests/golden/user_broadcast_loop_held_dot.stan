data {
  int<lower=0> N;
  int<lower=0> N_vs;
  int<lower=0> N_y;
  array[N] int k;
  array[N] int ns;
  vector[N_vs] vs;
  vector[N_y] y;
  vector[N_y] xs;
}
parameters {
  real<lower=0> b;
}
model {
  b ~ exponential(1.0);
  for (i in 1:N) {
    k[i] ~ binomial_logit(ns[i] + 1, sum(vs .* vs) * b);
  }
  for (i in 1:N_y) {
    y[i] ~ normal(xs[i] > 1.0 ? xs[i] : -xs[i], sum(vs .* vs) * b);
  }
}
