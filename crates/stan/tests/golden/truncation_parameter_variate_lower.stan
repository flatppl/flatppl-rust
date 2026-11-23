data {
  real y;
}
parameters {
  real a;
  real<lower=0> r;
  real<lower=0> z;
}
model {
  a ~ std_normal();
  r ~ exponential(1.0);
  if (z < fmax(a, 0)) {
    target += negative_infinity();
  } else {
    z ~ gamma(2.0, r);
    target += -(a <= 0 ? 0 : gamma_lccdf(a | 2.0, r));
  }
  y ~ normal(z, 1.0);
}
