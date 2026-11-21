data {
  int<lower=0> N;
  int<lower=0> N_R_obs_2;
  matrix[N, N_R_obs_2] R_obs;
}
transformed data {
  int d = dims(R_obs)[1];
  if (N_R_obs_2 != d) {
    reject("the size of R_obs is ", N_R_obs_2, ", but its law gives ", d);
  }
}
parameters {
  real<lower=0> eta;
}
model {
  eta ~ exponential(1);
  R_obs ~ lkj_corr(eta);
}
