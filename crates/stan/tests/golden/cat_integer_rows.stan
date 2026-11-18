data {
  int<lower=0> N;
  int<lower=0> J;
  array[N, J] int X;
}
transformed data {
  array[N + 3, J] int counts = append_array(append_array(X, rep_array(rep_array(0, J), 2)), {{4, 4, 4}});
}
parameters {
  real<lower=0> lam;
}
model {
  lam ~ exponential(1);
  to_array_1d(counts) ~ poisson(lam);
}
