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
  vector[3] x_obs;
}
transformed data {
  vector[3] y = x_obs * 2;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  if (min(y) < -1.0 || max(y) > 2.0) {
    target += negative_infinity();
  } else {
    y ~ normal(mu, 1);
    target += -3 * normal_interval_lmass(-1.0, 2.0, mu, 1);
  }
}
