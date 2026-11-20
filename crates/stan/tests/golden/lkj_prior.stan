data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  corr_matrix[3] C;
}
model {
  C ~ lkj_corr(2.0);
  y ~ multi_normal([0.0, 0.0, 0.0]', C);
}
