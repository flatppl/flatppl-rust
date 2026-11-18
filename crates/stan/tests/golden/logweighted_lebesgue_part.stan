data {
  int<lower=0> N_node1;
  int<lower=0> N_node2;
  int<lower=0> N;
  array[N_node1] int node1;
  array[N_node2] int node2;
  array[N] int y_obs;
}
parameters {
  real beta0;
  vector[N] phi;
}
model {
  beta0 ~ std_normal();
  target += -0.5 * sum((phi[node1] - phi[node2]) .^ 2) + normal_lpdf(sum(phi) | 0, 0.001 * N);
  y_obs ~ poisson_log(beta0 + phi);
}
