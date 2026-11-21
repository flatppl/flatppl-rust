data {
  real z;
}
parameters {
  real<lower=0, upper=2> t;
  real<lower=0> u;
}
model {
  inv(t / 2) - 1 ~ gamma(3, 2);
  target += -log(2);
  target += -2 * log(abs(t / 2));
  inv(u) ~ gamma(2, 1);
  target += -2 * log(abs(u));
  z ~ normal(t, u);
}
