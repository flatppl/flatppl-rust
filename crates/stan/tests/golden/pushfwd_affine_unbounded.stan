data {
  int<lower=0> N;
  vector[N] cs;
  real z;
}
transformed data {
  if (cs[2] == 0) {
    reject("pushfwd of rho: the scale cs[2] is 0, so the map is no bijection");
  }
}
parameters {
  real rho;
}
model {
  (rho - cs[1]) / cs[2] ~ normal(0.3, 2);
  target += -log(abs(cs[2]));
  z ~ normal(rho, 1);
}
