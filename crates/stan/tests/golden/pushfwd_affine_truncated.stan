data {
  real z;
}
parameters {
  real<lower=-3, upper=3> rho;
}
model {
  (rho + 1) / 2 ~ std_normal() T[-1, 2];
  target += -log(2);
  z ~ normal(rho, 1);
}
