functions {
  real pareto_interval_lmass(real lo, real hi, real p1, real p2) {
    real a = fmax(lo, p1);
    real b = fmax(hi, p1);
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = pareto_lccdf(a | p1, p2);
    if (la < log(0.5)) {
      return log_diff_exp(la, pareto_lccdf(b | p1, p2));
    }
    return log_diff_exp(pareto_lcdf(b | p1, p2), pareto_lcdf(a | p1, p2));
  }
}
data {
  real point;
  real ym;
  real lo;
  real hi;
}
model {
  if (point < fmax(lo, ym) || point > hi) {
    target += negative_infinity();
  } else {
    point ~ pareto(ym, 3.0);
    target += -pareto_interval_lmass(lo, hi, ym, 3.0);
  }
}
