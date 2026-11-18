data {
  int<lower=0> N;
  matrix[N, 2] A;
}
parameters {
  array[2] vector[2] w;
}
model {
  matrix[2, 2] stacked;
  for (i in 1:2) {
    stacked[i] = w[i]';
  }
  matrix[N, 2] lw_site = A * stacked';
  vector[N] lw_agg = rows_dot_self(lw_site);

  for (i in 1:2) {
    w[i] ~ std_normal();
  }
  target += sum(lw_agg);
}
