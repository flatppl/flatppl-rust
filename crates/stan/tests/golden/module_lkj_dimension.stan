data {
  int<lower=0> dim;
  matrix[dim, dim] R;
}
parameters {
  real<lower=0> eta;
}
model {
  eta ~ exponential(1);
  R ~ lkj_corr(eta);
}
