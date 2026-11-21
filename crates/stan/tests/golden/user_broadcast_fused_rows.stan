data {
  int<lower=0> N;
  int<lower=0> N_m;
  array[N] int k;
  vector[N] xs;
  array[N_m] int m;
  array[N_m] int ns;
  array[N] int c;
}
parameters {
  real b;
}
model {
  b ~ std_normal();
  k ~ poisson_log(xs);
  m ~ binomial_logit(ns, b);
  c ~ bernoulli_logit(b);
}
