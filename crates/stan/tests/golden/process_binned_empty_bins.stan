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
  real flatppl_binned_poisson_log_lpmf(array[] int n, vector lm) {
    real lp = 0;
    for (k in 1:num_elements(n)) {
      if (lm[k] == negative_infinity()) {
        lp += n[k] == 0 ? 0.0 : negative_infinity();
      } else {
        lp += poisson_log_lpmf(n[k] | lm[k]);
      }
    }
    return lp;
  }
}
data {
  int<lower=0> N;
  array[N] int y;
}
parameters {
  real mu;
  real a;
}
model {
  y ~ flatppl_binned_poisson_log([fmax(-2, a) < fmin(-1, a + 1) ? flatppl_normal_lmass(fmax(-2, a), fmin(-1, a + 1), mu, 1) : negative_infinity(), fmax(-1, a) < fmin(0, a + 1) ? flatppl_normal_lmass(fmax(-1, a), fmin(0, a + 1), mu, 1) : negative_infinity(), fmax(0, a) < fmin(1, a + 1) ? flatppl_normal_lmass(fmax(0, a), fmin(1, a + 1), mu, 1) : negative_infinity()]');
}
