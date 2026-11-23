functions {
  real pareto_interval_lmass(real lo, real hi, real p1, real p2) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= p1) {
      return negative_infinity();
    }
    if (lo <= p1) {
      return pareto_lcdf(hi | p1, p2);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = pareto_lccdf(lo | p1, p2);
    if (la < log(0.5)) {
      return log_diff_exp(la, pareto_lccdf(hi | p1, p2));
    }
    return log_diff_exp(pareto_lcdf(hi | p1, p2), pareto_lcdf(lo | p1, p2));
  }
  real flatppl_nonzero_lmass(real lmass) {
    if (lmass == negative_infinity()) {
      reject("normalize of a measure of zero mass: spec section 06 takes normalize of a measure whose total mass is finite and nonzero");
    }
    return lmass;
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
    target += -flatppl_nonzero_lmass(pareto_interval_lmass(lo, hi, ym, 3.0));
  }
}
