parameters {
  matrix[3, 3] c_obs;
  real<lower=0> eta;
}
model {
  c_obs ~ lkj_corr(eta);
}
