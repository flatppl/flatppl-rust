data {
  int<lower=0> N;
  vector[N] cs;
  real z;
}
parameters {
  real rho;
}
model {
  (rho - cs[1]) / cs[2] ~ normal(0.3, 2);
  target += -log(abs(cs[2]));
  z ~ normal(rho, 1);
}
