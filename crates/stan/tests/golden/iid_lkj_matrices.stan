data {
  array[2] matrix[2, 2] R;
}
parameters {
  real<lower=0> eta;
}
model {
  eta ~ exponential(1);
  for (i in 1:2) {
    R[i] ~ lkj_corr(eta);
  }
}
