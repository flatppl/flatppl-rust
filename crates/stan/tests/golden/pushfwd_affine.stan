data {
  real z;
}
parameters {
  real<lower=-1, upper=1> rho;
}
model {
  (rho + 1) / 2 ~ beta(2, 2);
  target += -log(2);
  z ~ normal(rho, 1);
}
