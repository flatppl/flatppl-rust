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
  int<lower=0> N;
  vector[N] obs;
}
transformed data {
  real lmass = flatppl_normal_lmass(0, 10, 5, 0.8);
}
parameters {
  real<lower=0> s;
  real<lower=0> b;
}
model {
  s ~ gamma(5.0, 0.5);
  b ~ gamma(8.0, 0.5);
  target += -(s + b);
  for (i in 1:num_elements(obs)) {
    target += log_sum_exp(log(s) + ((obs[i] < 0 || obs[i] > 10 ? negative_infinity() : normal_lpdf(obs[i] | 5.0, 0.8)) - lmass), log(b) + (log(0.1) + (obs[i] < 0 || obs[i] > 10 ? negative_infinity() : 0)));
  }
}
