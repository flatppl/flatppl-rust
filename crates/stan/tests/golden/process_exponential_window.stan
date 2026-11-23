data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real<lower=0> r;
  real<lower=0> n;
}
model {
  real lmass = exponential_lcdf(10 | r);

  target += num_elements(y) * log(n) + ((min(y) < 0 || max(y) > 10 ? negative_infinity() : exponential_lpdf(y | r)) - num_elements(y) * lmass) - n;
}
