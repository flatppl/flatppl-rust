data {
  real z;
}
parameters {
  real<lower=1> s;
}
model {
  log(expm1((s - 1) / 2)) ~ normal(0.1, 0.7);
  target += -log(2);
  target += (s - 1) / 2;
  target += -log(expm1((s - 1) / 2));
  z ~ normal(s, 1);
}
