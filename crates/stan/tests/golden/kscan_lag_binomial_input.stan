data {
  int<lower=0> N;
  array[N] int k;
  array[N] int ns;
}
transformed data {
  vector[N] k_lag = append_row(1, to_vector(k[1:(N - 1)]));
}
parameters {
  real b;
}
model {
  b ~ std_normal();
  k ~ binomial_logit(ns, 0.2 * b * k_lag - 0.5);
}
