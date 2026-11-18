data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
}
parameters {
  vector[2] b;
  vector<lower=0>[1] c;
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + b[1] * x + b[2], c[1]);
}
