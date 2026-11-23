functions {
  real gamma_interval_lmass(real lo, real hi, real p1, real p2) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= 0) {
      return negative_infinity();
    }
    if (lo <= 0) {
      return gamma_lcdf(hi | p1, p2);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = gamma_lccdf(lo | p1, p2);
    if (la < log(0.5)) {
      return log_diff_exp(la, gamma_lccdf(hi | p1, p2));
    }
    return log_diff_exp(gamma_lcdf(hi | p1, p2), gamma_lcdf(lo | p1, p2));
  }
}
data {
  real y;
}
parameters {
  vector[2] theta;
}
model {
  real s = theta[1];
  real v = theta[2];

  if (s < 0.5 || s > 4.0) {
    target += negative_infinity();
  } else {
    s ~ gamma(2.0, 1.0);
    target += -gamma_interval_lmass(0.5, 4.0, 2.0, 1.0);
  }
  if (v < -pi() || v > pi()) {
    target += negative_infinity();
  } else {
    v ~ von_mises(0.0, 2.0);
  }
  y ~ normal(s + v, 1.0);
}
