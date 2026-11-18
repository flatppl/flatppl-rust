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
    target += log_mix(w, (y_obs[i] < -1.0 || y_obs[i] > 2.0 ? negative_infinity() : normal_lpdf(y_obs[i] | mu, 1) - normal_interval_lmass(-1.0, 2.0, mu, 1)), normal_lpdf(y_obs[i] | 3, 1));
  }
}
