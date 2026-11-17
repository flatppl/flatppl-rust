data {
  int<lower=0> K;
  int<lower=0> N;
  array[N] int w_obs;
  array[N] int doc;
}
parameters {
  array[K] simplex[3] theta;
}
model {
  for (i in 1:K) {
    theta[i] ~ dirichlet(rep_vector(1.0, 3));
  }
  for (i in 1:N) {
    w_obs[i] ~ categorical(theta[doc][i]);
  }
}
