data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  cholesky_factor_corr[2] L_corr;
}
model {
  L_corr ~ lkj_corr_cholesky(1);
  y ~ multi_normal([0.0, 0.0]', L_corr * L_corr');
}
