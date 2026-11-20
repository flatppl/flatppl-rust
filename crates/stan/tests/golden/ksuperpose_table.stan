data {
  vector[4] y;
}
parameters {
  simplex[2] weights;
}
model {
  weights ~ dirichlet([2.0, 2.0]');
  for (i in 1:4) {
    {
      vector[2] lps;
      for (j in 1:2) {
        lps[j] = normal_lpdf(y[i] | ([-2.0, 2.0]')[j], ([0.7, 0.9]')[j]);
      }
      target += log_sum_exp(log(weights) + lps);
    }
  }
}
