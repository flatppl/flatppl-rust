data {
  real z;
}
parameters {
  real<upper=0> r;
}
model {
  -r ~ exponential(2);
  z ~ normal(r, 1);
}
