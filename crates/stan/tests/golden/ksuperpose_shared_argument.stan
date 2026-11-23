data {
  real y;
}
parameters {
  real mu;
  real nu;
}
model {
  {
    vector[2] lps;
    for (i in 1:2) {
      lps[i] = normal_lpdf(y | ([3.0, 4.0]')[i] - 2.0 * (mu - nu), 1.0);
    }
    target += log_sum_exp(log([0.3, 0.7]') + lps);
  }
}
