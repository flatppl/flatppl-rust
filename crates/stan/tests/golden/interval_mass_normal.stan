functions {
  real normal_interval_lmass(real lo, real hi, real p1, real p2) {
    // Above the median the upper tail comes from its mirror image: normal_lccdf is -inf far out.
    real la = normal_lcdf(lo | p1, p2);
    if (la > log(0.5)) {
      return log_diff_exp(normal_lcdf(2 * p1 - lo | p1, p2), normal_lcdf(2 * p1 - hi | p1, p2));
    }
    return log_diff_exp(normal_lcdf(hi | p1, p2), la);
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
    target += -normal_interval_lmass(lo, hi, mu, sigma);
  }
}
