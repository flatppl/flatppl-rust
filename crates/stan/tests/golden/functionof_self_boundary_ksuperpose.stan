data {
  vector[3] y;
}
parameters {
  simplex[2] w;
}
model {
  w ~ dirichlet([1.0, 1.0]');
  for (i in 1:3) {
    {
      vector[2] lps;
      for (j in 1:2) {
        lps[j] = normal_lpdf(y[i] | 0.5 * ([3.0, 4.0]')[j] + 1, 1);
      }
      target += log_mix(w, lps);
    }
  }
}
