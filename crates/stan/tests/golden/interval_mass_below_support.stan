functions {
  real exponential_interval_lmass(real lo, real hi, real p1) {
    real a = fmax(lo, 0);
    real b = fmax(hi, 0);
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = exponential_lccdf(a | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, exponential_lccdf(b | p1));
    }
    return log_diff_exp(exponential_lcdf(b | p1), exponential_lcdf(a | p1));
  }
}
data {
  real y;
}
model {
  if (y < 0 || y > 2.0) {
    target += negative_infinity();
  } else {
    y ~ exponential(1.0);
    target += -exponential_interval_lmass(-1.0, 2.0, 1.0);
  }
}
