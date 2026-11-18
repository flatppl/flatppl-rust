functions {
  real normal_interval_lmass(real lo, real hi, real p1, real p2) {
    // Above the median the upper tail comes from its mirror image: normal_lccdf is -inf far out.
    real la = normal_lcdf(lo | p1, p2);
    if (la > log(0.5)) {
      return log_diff_exp(normal_lcdf(2 * p1 - lo | p1, p2), normal_lcdf(2 * p1 - hi | p1, p2));
    }
    return log_diff_exp(normal_lcdf(hi | p1, p2), la);
  }
  real poisson_interval_lmass(int lo, int hi, real p1) {
    int a = lo - 1;
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = poisson_lccdf(a | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, poisson_lccdf(hi | p1));
    }
    return log_diff_exp(poisson_lcdf(hi | p1), poisson_lcdf(a | p1));
  }
}
data {
  vector[3] y;
  int k;
}
parameters {
  real mu;
  real<lower=0> lam;
}
model {
  mu ~ std_normal();
  lam ~ exponential(1);
  if (min(y) < -1.0 || max(y) > 2.0) {
    target += negative_infinity();
  } else {
    y ~ normal(mu, 1);
    target += -3 * normal_interval_lmass(-1.0, 2.0, mu, 1);
  }
  if (k < 1 || k > 5) {
    target += negative_infinity();
  } else {
    k ~ poisson(lam);
    target += -poisson_interval_lmass(1, 5, lam);
  }
}
