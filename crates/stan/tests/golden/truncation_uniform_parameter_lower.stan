functions {
  real uniform_interval_lmass(real lo, real hi, real p1, real p2) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= p1) {
      return negative_infinity();
    }
    if (lo >= p2) {
      return negative_infinity();
    }
    if (lo <= p1 && hi >= p2) {
      return 0;
    }
    if (lo <= p1) {
      return uniform_lcdf(hi | p1, p2);
    }
    if (hi >= p2) {
      return uniform_lccdf(lo | p1, p2);
    }
    return log_diff_exp(uniform_lcdf(hi | p1, p2), uniform_lcdf(lo | p1, p2));
  }
  real flatppl_nonzero_lmass(real lmass) {
    if (lmass == negative_infinity()) {
      reject("normalize of a measure of zero mass: spec section 06 takes normalize of a measure whose total mass is finite and nonzero");
    }
    return lmass;
  }
}
data {
  real y;
}
parameters {
  real<lower=0> p;
  real a;
}
model {
  if (y < fmax(a, 0.0) || y > fmin(1.2, p)) {
    target += negative_infinity();
  } else {
    y ~ uniform(0.0, p);
    target += -flatppl_nonzero_lmass(uniform_interval_lmass(a, 1.2, 0.0, p));
  }
}
