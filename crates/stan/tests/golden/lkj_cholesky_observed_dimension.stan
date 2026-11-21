data {
  int<lower=0> dim;
  matrix[dim, dim] L;
}
parameters {
  real<lower=0> eta;
}
model {
  eta ~ exponential(1);
  if (max(abs(rows_dot_self(L) - 1)) > 1e-8) {
    target += negative_infinity();
  } else {
    L ~ lkj_corr_cholesky(eta);
  }
}
