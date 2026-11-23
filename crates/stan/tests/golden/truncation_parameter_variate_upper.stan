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
  real c;
  real<lower=0> r;
  real<lower=0, upper=c> z;
}
model {
  c ~ normal(3.0, 1.0);
  r ~ exponential(1.0);
  z ~ gamma(2.0, r);
  target += -flatppl_nonzero_lmass(gamma_interval_lmass(0, c, 2.0, r));
  y ~ normal(z, 1.0);
}
