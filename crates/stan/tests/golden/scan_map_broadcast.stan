data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x_obs;
}
parameters {
  real a;
}
model {
  vector[N] scan = a * x_obs + 1;

  a ~ std_normal();
  y_obs ~ normal(scan, 1);
}
