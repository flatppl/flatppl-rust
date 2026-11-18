functions {
  real flatppl_argus_lpdf(real x, real m0, real c, real p) {
    if (x <= 0 || x >= m0) return negative_infinity();
    real u = 1 - square(x / m0);
    // log of the integral of u^p exp(c u) over (0, 1)
    real log_int;
    // The integral is 1F1(p + 1; p + 2; c) / (p + 1). For c < 0 use Kummer's transform so the
    // series has positive terms. gamma_p is avoided: its gradient in the shape argument is off by
    // 3e-7 relative at (1.5, 20) in Stan 2.39. No c == 0 branch: a constant would zero d/dc.
    if (-c > p + 1 && -c + lgamma(p + 2) - (p + 1) * log(-c) >= 700) {
      // For -c > p + 1 the Kummer series 1F1(1; p + 2; -c) grows like
      // exp(-c) Gamma(p + 2) / (-c)^(p + 1) and overflows past exp(700); the integral
      // is then Gamma(p + 1) P(p + 1, -c) / (-c)^(p + 1). For -c <= p + 1 it stays small.
      log_int = lgamma(p + 1) + log(gamma_p(p + 1, -c)) - (p + 1) * log(-c);
    } else if (c < 0) {
      log_int = c + log(hypergeometric_pFq([1]', [p + 2]', -c)) - log1p(p);
    } else {
      log_int = log(hypergeometric_pFq([p + 1]', [p + 2]', c)) - log1p(p);
    }
    return log(x) + p * log(u) + c * u - (2 * log(m0) - log2() + log_int);
  }
  real flatppl_argus_lpdf(vector x, real m0, real c, real p) {
    int n = num_elements(x);
    if (n == 0) return 0;
    if (min(x) <= 0 || max(x) >= m0) return negative_infinity();
    vector[n] u = 1 - square(x / m0);
    real log_int;
    if (-c > p + 1 && -c + lgamma(p + 2) - (p + 1) * log(-c) >= 700) {
      // For -c > p + 1 the Kummer series 1F1(1; p + 2; -c) grows like
      // exp(-c) Gamma(p + 2) / (-c)^(p + 1) and overflows past exp(700); the integral
      // is then Gamma(p + 1) P(p + 1, -c) / (-c)^(p + 1). For -c <= p + 1 it stays small.
      log_int = lgamma(p + 1) + log(gamma_p(p + 1, -c)) - (p + 1) * log(-c);
    } else if (c < 0) {
      log_int = c + log(hypergeometric_pFq([1]', [p + 2]', -c)) - log1p(p);
    } else {
      log_int = log(hypergeometric_pFq([p + 1]', [p + 2]', c)) - log1p(p);
    }
    return sum(log(x) + p * log(u) + c * u) - n * (2 * log(m0) - log2() + log_int);
  }
}
data {
  vector[5] y_obs;
}
parameters {
  real c;
}
model {
  c ~ normal(-20, 5);
  y_obs ~ flatppl_argus(5.29, c, 0.5);
}
