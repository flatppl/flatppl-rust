functions {
  real gamma_interval_lmass(real lo, real hi, real p1, real p2) {
    real a = fmax(lo, 0);
    real b = fmax(hi, 0);
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = gamma_lccdf(a | p1, p2);
    if (la < log(0.5)) {
      return log_diff_exp(la, gamma_lccdf(b | p1, p2));
    }
    return log_diff_exp(gamma_lcdf(b | p1, p2), gamma_lcdf(a | p1, p2));
  }
}
data {
  real point;
  real rate;
  real lo;
  real hi;
}
model {
  if (point < fmax(lo, 0) || point > hi) {
    target += negative_infinity();
  } else {
    point ~ gamma(2.0, rate);
    target += -gamma_interval_lmass(lo, hi, 2.0, rate);
  }
}
