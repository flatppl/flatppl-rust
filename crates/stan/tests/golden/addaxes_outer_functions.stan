data {
  int<lower=0> N;
  int<lower=0> N_v;
  real z;
  vector[N] x;
  array[N] int k;
  vector[N_v] v;
}
transformed data {
  matrix[N, N_v] a = fmax(rep_matrix(x, N_v), rep_matrix(v', N)) + rep_matrix(to_vector(fmin(to_matrix(to_vector(k)), 1.5)), N_v);
  matrix[N, N_v] b = fmax(rep_matrix(x, N_v) + rep_matrix(v', N), 0.5);
  matrix[N, N_v] c = rep_matrix(to_vector(k), N_v) + rep_matrix(v', N);
}
parameters {
  real mu;
}
model {
  matrix[N, N_v] w = rep_matrix(to_row_vector(exp(to_matrix(v') * mu)), N) + rep_matrix(x, N_v);

  mu ~ std_normal();
  z ~ normal(mu + sum(a + b + c + w), 1);
}
