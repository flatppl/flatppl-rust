data {
  int<lower=0> N;
  int<lower=0> N_Lobs_2;
  real m;
  matrix[N, N_Lobs_2] Lobs;
}
model {
  if (max(abs(rows_dot_self(Lobs) - 1)) > 1e-8) {
    target += negative_infinity();
  } else {
    Lobs ~ lkj_corr_cholesky(2.0);
  }
  m ~ std_normal();
}
