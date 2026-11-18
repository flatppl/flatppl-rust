data {
  real x;
  real lo;
}
model {
  target += log_sum_exp([log(0.4) + (x < fmax(lo, 0) || x > 1 ? negative_infinity() : beta_lpdf(x | 2.0, 3.0) - log_diff_exp(beta_lcdf(1 | 2.0, 3.0), beta_lcdf(fmin(fmax(lo, 0), 1) | 2.0, 3.0))), log(0.6) + (x < fmax(lo, 0) ? negative_infinity() : gamma_lpdf(x | 2.0, 1.0) - gamma_lccdf(fmax(lo, 0) | 2.0, 1.0))]') - log(0.4 + 0.6);
}
