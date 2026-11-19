data {
  real y;
}
parameters {
  complex z;
}
model {
  target += -0.5 * norm(z);
  y ~ normal(get_real(z) + 3 * get_imag(z), 1);
}
