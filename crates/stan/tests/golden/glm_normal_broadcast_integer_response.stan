data {
  int<lower=0> N;
  int<lower=0> N_x_2;
  array[2, N] int y;
  matrix[N, N_x_2] x;
}
parameters {
  vector[2] a;
  vector[2] beta;
}
model {
  a ~ std_normal();
  beta ~ std_normal();
  for (i in 1:2) {
    to_vector(y[i]) ~ normal_id_glm({x, x}[i], a[i], {beta, beta}[i], 1.0);
  }
}
