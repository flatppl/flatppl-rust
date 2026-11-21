data {
  real a;
  real b;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  if (a <= 0) {
    target += negative_infinity();
  } else {
    log(a) ~ normal(mu, 1.0);
    target += -log(a);
  }
  if (b <= 0 || b >= 1) {
    target += negative_infinity();
  } else {
    logit(b) ~ normal(mu, 1.0);
    target += -log(b) - log1m(b);
  }
}
