data {
  real z;
}
parameters {
  real<lower=-1, upper=2> rho;
}
model {
  (2 - rho) / 3 ~ beta(2, 3);
  target += -log(3);
  z ~ normal(rho, 1);
}
