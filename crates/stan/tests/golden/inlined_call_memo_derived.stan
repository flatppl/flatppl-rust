data {
  int<lower=0> N;
  int<lower=0> K;
  real z;
  matrix[N, K] X;
}
parameters {
  vector[K] p;
}
transformed parameters {
  vector[N] f = X * p;
}
model {
  p ~ std_normal();
  z ~ normal(sum(f) + f[2], 1);
}
