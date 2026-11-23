functions {
  real beta_interval_lmass(real lo, real hi, real p1, real p2) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= 0) {
      return negative_infinity();
    }
    if (lo >= 1) {
      return negative_infinity();
    }
    if (lo <= 0 && hi >= 1) {
      return 0;
    }
    if (lo <= 0) {
      return beta_lcdf(hi | p1, p2);
    }
    if (hi >= 1) {
      return beta_lccdf(lo | p1, p2);
    }
    return log_diff_exp(beta_lcdf(hi | p1, p2), beta_lcdf(lo | p1, p2));
  }
  real flatppl_nonzero_lmass(real lmass) {
    if (lmass == negative_infinity()) {
      reject("normalize of a measure of zero mass: spec section 06 takes normalize of a measure whose total mass is finite and nonzero");
    }
    return lmass;
  }
}
data {
  int a;
  real y;
}
parameters {
  real<lower=0> p;
}
model {
  if (y < fmax(a, 0) || y > 0.9) {
    target += negative_infinity();
  } else {
    y ~ beta(p, 3.0);
    target += -flatppl_nonzero_lmass(beta_interval_lmass(a, 0.9, p, 3.0));
  }
}
