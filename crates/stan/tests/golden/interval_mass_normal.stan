functions {
  real flatppl_lphi(real z) {
    // log Phi(z) from erfc for z >= -37: Stan's normal_lcdf derivative is an approximation.
    if (z > 0) {
      return log1m(0.5 * erfc(z / sqrt2()));
    }
    if (z >= -37) {
      return log(erfc(-z / sqrt2())) - log2();
    }
    return std_normal_lcdf(z);
  }
  real flatppl_normal_lmass(real lo, real hi, real mu, real sigma) {
    // A window whose ends cross at run time has no mass (`log_diff_exp` of an equal
    // or crossed pair is NaN). An interval above the median is taken as its mirror
    // image below it.
    if (lo >= hi) {
      return negative_infinity();
    }
    real a = (lo - mu) / sigma;
    real b = (hi - mu) / sigma;
    if (a > 0) {
      return log_diff_exp(flatppl_lphi(-a), flatppl_lphi(-b));
    }
    return log_diff_exp(flatppl_lphi(b), flatppl_lphi(a));
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
  real mu;
  real sigma;
  real lo;
  real hi;
}
model {
  if (point < lo || point > hi) {
    target += negative_infinity();
  } else {
    point ~ normal(mu, sigma);
    target += -flatppl_nonzero_lmass(flatppl_normal_lmass(lo, hi, mu, sigma));
  }
}
