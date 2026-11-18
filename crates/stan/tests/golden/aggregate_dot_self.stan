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
  matrix[N, 2] rows_site = A * stacked';
  vector[N] rows_agg = rows_dot_self(rows_site);
  matrix[2, 2] stacked_2;
  for (i in 1:2) {
    stacked_2[i] = w[i]';
  }
  matrix[N, 2] cols_site = A * stacked_2';
  vector[2] cols_agg = columns_dot_self(cols_site)';

  for (i in 1:2) {
    w[i] ~ std_normal();
  }
  target += sum(rows_agg) - 0.5 * sum(cols_agg .* [1.0, 2.0]');
}
