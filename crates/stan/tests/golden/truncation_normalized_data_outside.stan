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
  real poisson_interval_lmass(int lo, int hi, real p1) {
    // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.
    if (lo > hi) {
      return negative_infinity();
    }
    if (hi < 0) {
      return negative_infinity();
    }
    if (lo <= 0) {
      return poisson_lcdf(hi | p1);
    }
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = poisson_lccdf(lo - 1 | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, poisson_lccdf(hi | p1));
    }
    return log_diff_exp(poisson_lcdf(hi | p1), poisson_lcdf(lo - 1 | p1));
  }
}
data {
  vector[3] y;
  int k;
}
parameters {
  real mu;
  real<lower=0> lam;
}
model {
  mu ~ std_normal();
  lam ~ exponential(1);
  if (min(y) < -1.0 || max(y) > 2.0) {
    target += negative_infinity();
  } else {
    y ~ normal(mu, 1);
    target += -3 * flatppl_normal_lmass(-1.0, 2.0, mu, 1);
  }
  if (k < 1 || k > 5) {
    target += negative_infinity();
  } else {
    k ~ poisson(lam);
    target += -poisson_interval_lmass(1, 5, lam);
  }
}
