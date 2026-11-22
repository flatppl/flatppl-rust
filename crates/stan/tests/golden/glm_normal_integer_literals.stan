data {
  int<lower=0> N;
  int a;
  array[2] int beta;
  array[N] int y;
}
model {
  a ~ std_normal();
  beta ~ std_normal();
  to_vector(y) ~ normal_id_glm(to_matrix(append_row(append_row(to_vector({1, 0})', to_vector({0, 1})'), to_vector({1, 1})')), a, to_vector(beta), 1);
}
