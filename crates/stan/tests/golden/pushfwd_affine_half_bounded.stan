data {
  real z;
}
parameters {
  real<upper=4> rho;
}
model {
  4 - rho ~ exponential(1.5);
  z ~ normal(rho, 1);
}
