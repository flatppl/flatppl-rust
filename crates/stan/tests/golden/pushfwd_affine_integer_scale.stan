data {
  int<lower=0> N;
  array[N] int ks;
  real z;
}
parameters {
  real rho;
}
model {
  rho / (1.0 * ks[2] / 2) ~ std_normal();
  target += -log(abs(1.0 * ks[2] / 2));
  z ~ normal(rho, 1);
}
