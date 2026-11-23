functions {
  real exponential_interval_lmass(real lo, real hi, real p1) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo >= hi) {
      return negative_infinity();
    }
    if (hi <= 0) {
      return negative_infinity();
    }
    if (lo <= 0) {
      return exponential_lcdf(hi | p1);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = exponential_lccdf(lo | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, exponential_lccdf(hi | p1));
    }
    return log_diff_exp(exponential_lcdf(hi | p1), exponential_lcdf(lo | p1));
  }
  real flatppl_nonzero_lmass(real lmass) {
    if (lmass == negative_infinity()) {
      reject("normalize of a measure of zero mass: spec section 06 takes normalize of a measure whose total mass is finite and nonzero");
    }
    return lmass;
  }
}
data {
  int<lower=0> N;
  int c;
  vector[N] y;
}
parameters {
  real<lower=0> r;
}
model {
  real lmass = flatppl_nonzero_lmass(exponential_interval_lmass(0, c, r));

  target += (max(y) > c ? negative_infinity() : exponential_lpdf(y | r)) - num_elements(y) * lmass - 1;
}
