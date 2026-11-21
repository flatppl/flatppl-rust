data {
  real a;
  real b;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  if (a < 0) {
    target += negative_infinity();
  } else {
    square(a) ~ inv_gamma(5, 5);
    target += log(2) + log(a);
  }
  if (b <= 0) {
    target += negative_infinity();
  } else {
    log(b) ~ normal(mu, 0.8);
    target += -log(b);
  }
}
