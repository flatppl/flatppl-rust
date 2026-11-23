data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real<lower=0> n;
}
model {
  target += num_elements(y) * log(n) + std_normal_lpdf(y) - n;
}
