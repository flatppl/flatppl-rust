data {
  int k;
}
model {
  target += log_sum_exp([log(0.3) + (k < 1 ? negative_infinity() : poisson_lpmf(k | 3.0)), log(0.7) + (k < 0 ? negative_infinity() : neg_binomial_lpmf(k | 1, 0.4 / (1 - 0.4)))]') - log_sum_exp([log(0.3) + poisson_lccdf(0 | 3.0), log(0.7)]');
}
