data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real m0;
}
model {
  m0 ~ std_normal();
  y ~ normal([m0, m0 + 1.0]', [0.7, 0.9]');
}
