data {
  real m1_c;
  real m2_c;
}
parameters {
  real x;
}
model {
  x ~ normal(m1_c, 1.0);
  x ~ normal(m2_c, 1.0);
}
