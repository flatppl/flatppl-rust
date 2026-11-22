data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int y;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  real<lower=0, upper=1> w;
  vector[2] beta;
}
model {
  w ~ beta(2, 2);
  beta ~ std_normal();
  target += log_mix(w, normal_id_glm_lpdf(to_vector(y) | x, 0.0, beta, 1.0), normal_id_glm_lpdf(to_vector(y) | x, 1.0, beta, 2.0));
}
