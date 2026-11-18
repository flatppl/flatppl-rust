data {
  array[2, 3] int observed;
}
parameters {
  real<lower=0> r;
}
model {
  to_array_1d(observed) ~ poisson(r);
}
