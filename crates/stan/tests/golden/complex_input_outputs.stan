parameters {
  complex z;
  real y;
}
model {
  y ~ normal(get_real(z) + 3 * get_imag(z), 1);
}
