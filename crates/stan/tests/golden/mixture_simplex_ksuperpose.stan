data {
  int<lower=0> K;
  int<lower=0> N;
  vector[N] y_obs;
}
parameters {
  simplex[K] theta;
  vector[K] mu;
  vector<lower=0>[K] sigma;
}
model {
  theta ~ dirichlet(ones_vector(K));
  mu ~ normal(0, 5);
  sigma ~ exponential(1.0);
  for (i in 1:N) {
    {
      vector[K] lps;
      for (j in 1:K) {
        lps[j] = normal_lpdf(y_obs[i] | mu[j], sigma[j]);
      }
      target += log_mix(theta, lps);
    }
  }
}
