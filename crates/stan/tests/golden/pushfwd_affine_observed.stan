data {
  real y;
}
parameters {
  real<lower=0> a;
}
model {
  a ~ gamma(2, 2);
  (y + 1) / 2 ~ beta(a, 2);
  target += -log(2);
}
