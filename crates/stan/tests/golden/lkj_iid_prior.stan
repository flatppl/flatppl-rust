data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  array[2] corr_matrix[3] C;
}
model {
  for (i in 1:2) {
    C[i] ~ lkj_corr(2.0);
  }
  y ~ multi_normal([0.0, 0.0, 0.0]', C[1]);
}
