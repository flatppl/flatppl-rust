functions {
  real flatppl_crystal_ball_lpdf(real x, real m0, real sigma, real alpha, real n) {
    real a = abs(alpha);
    real t = (x - m0) / sigma;
    real log_tail = log(n) - log(a) - log(n - 1) - 0.5 * square(a);
    real log_core = 0.5 * log(pi() / 2) + log1p(erf(a / sqrt2()));
    real log_norm = log(sigma) + log_sum_exp(log_tail, log_core);
    if (t < -a) return n * log(n / a) - 0.5 * square(a) - n * log(n / a - a - t) - log_norm;
    return -0.5 * square(t) - log_norm;
  }
  real flatppl_crystal_ball_lpdf(vector x, real m0, real sigma, real alpha, real n) {
    // Piecewise in x: the normalizer once, the branch per element.
    real a = abs(alpha);
    real log_tail = log(n) - log(a) - log(n - 1) - 0.5 * square(a);
    real log_core = 0.5 * log(pi() / 2) + log1p(erf(a / sqrt2()));
    real lp = -num_elements(x) * (log(sigma) + log_sum_exp(log_tail, log_core));
    for (i in 1:num_elements(x)) {
      real t = (x[i] - m0) / sigma;
      lp += t < -a ? n * log(n / a) - 0.5 * square(a) - n * log(n / a - a - t) : -0.5 * square(t);
    }
    return lp;
  }
}
data {
  vector[6] y_obs;
}
parameters {
  real<lower=0> s;
}
model {
  s ~ exponential(200.0);
  y_obs ~ flatppl_crystal_ball(5.279, s, 1.5, 3.0);
}
