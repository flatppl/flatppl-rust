data {
  int<lower=0> N;
  array[N] int ks;
  real z;
}
parameters {
  real<lower=1.0 * ks[1] / 2, upper=0.5 + 1.0 * ks[1] / 2> rho;
}
model {
  (rho - 1.0 * ks[1] / 2) / 0.5 ~ beta(2, 2);
  target += -log(0.5);
  z ~ normal(rho, 1);
}
