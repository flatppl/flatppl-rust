data {
  real m;
  matrix[3, 3] Lobs;
}
model {
  if (max(abs(rows_dot_self(Lobs) - 1)) > 1e-8) {
    target += negative_infinity();
  } else {
    Lobs ~ lkj_corr_cholesky(2.0);
  }
  m ~ std_normal();
}
