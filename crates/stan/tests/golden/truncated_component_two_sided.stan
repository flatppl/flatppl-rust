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
}
data {
  vector[3] y_obs;
}
parameters {
  real<lower=0, upper=1> w;
  real mu;
}
model {
  w ~ beta(2, 2);
  mu ~ std_normal();
  for (i in 1:3) {
    target += log_mix(w, (y_obs[i] < -1.0 || y_obs[i] > 2.0 ? negative_infinity() : normal_lpdf(y_obs[i] | mu, 1) - flatppl_normal_lmass(-1.0, 2.0, mu, 1)), normal_lpdf(y_obs[i] | 3, 1));
  }
}
