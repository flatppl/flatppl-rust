data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
}
parameters {
  vector[K] p;
}
model {
  vector[N] f = X * p;
  vector[N] f_2 = X * (2.0 * p);

  p ~ std_normal();
  target += sum(f) + f[1] * f[3] + sum(p) + sum(p) + sum(f_2);
}
