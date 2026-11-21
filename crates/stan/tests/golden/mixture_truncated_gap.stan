functions {
  real exponential_interval_lmass(real lo, real hi, real p1) {
    real a = fmax(lo, 0);
    real b = fmax(hi, 0);
    // Above the median the survival pair keeps the precision that the lcdf pair loses.
    real la = exponential_lccdf(a | p1);
    if (la < log(0.5)) {
      return log_diff_exp(la, exponential_lccdf(b | p1));
    }
    return log_diff_exp(exponential_lcdf(b | p1), exponential_lcdf(a | p1));
  }
}
data {
  vector[3] y_obs;
}
parameters {
  real mu;
  real<lower=0> r;
}
model {
  mu ~ std_normal();
  r ~ exponential(1);
  for (i in 1:3) {
    target += log_sum_exp([log(0.4) + normal_lpdf(y_obs[i] | mu, 1.0), log(0.6) + (y_obs[i] < 0 || y_obs[i] > 3.0 ? negative_infinity() : exponential_lpdf(y_obs[i] | r) - exponential_interval_lmass(-1.0, 3.0, r))]');
  }
}
