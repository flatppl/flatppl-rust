data {
  int<lower=0> K;
  int<lower=0> N_w_obs;
  array[N_w_obs] int w_obs;
  int<lower=0> N_doc;
  array[N_doc] int doc;
}
parameters {
  array[K] simplex[3] theta;
}
model {
  for (i in 1:K) {
    theta[i] ~ dirichlet(ones_vector(3));
  }
  for (i in 1:N_w_obs) {
    w_obs[i] ~ categorical(theta[doc[i]]);
  }
}
