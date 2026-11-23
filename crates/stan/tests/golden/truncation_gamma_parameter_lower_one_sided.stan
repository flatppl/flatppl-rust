data {
  real y;
}
parameters {
  real<lower=0> r;
  real a;
}
model {
  if (y < fmax(a, 0)) {
    target += negative_infinity();
  } else {
    y ~ gamma(2.0, r);
    target += -(a <= 0 ? 0 : gamma_lccdf(a | 2.0, r));
  }
}
