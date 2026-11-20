data {
  real y;
}
parameters {
  real q;
}
model {
  matrix[3, 3] L = append_row(append_row([1.0, 0.0, 0.0], [q, 0.8, 0.0]), [0.0, 0.6, 0.8]);

  if (max(abs(rows_dot_self(L) - 1)) > 1e-8) {
    target += negative_infinity();
  } else {
    L ~ lkj_corr_cholesky(2.0);
  }
  y ~ std_normal();
}
