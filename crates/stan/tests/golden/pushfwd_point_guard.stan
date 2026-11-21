data {
  real z;
}
parameters {
  real m;
  real s;
  real r;
}
model {
  m ~ std_normal();
  if (s < 0) {
    target += negative_infinity();
  } else {
    square(s) ~ inv_gamma(5, 5);
    target += log(2) + log(s);
  }
  if (r <= 0) {
    target += negative_infinity();
  } else {
    log(r) ~ normal(0.3, 0.8);
    target += -log(r);
  }
  z ~ normal(m, 1);
}
