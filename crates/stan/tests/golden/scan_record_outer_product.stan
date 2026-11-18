data {
  int<lower=0> N_v_obs;
  real z;
  int<lower=0> N;
  int<lower=0> M;
  vector[N_v_obs] v_obs;
}
parameters {
  real r;
}
model {
  matrix[N, M] a = rep_matrix(0.0, N, M);
  vector[M] b = rep_vector(1.0, M);
  for (t in 1:N) {
    a = a + one_hot_vector(N, t) * (r * b)';
    b = b + v_obs;
  }

  r ~ std_normal();
  z ~ normal(sum(a), 1);
}
