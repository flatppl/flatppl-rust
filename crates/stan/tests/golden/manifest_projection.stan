data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
  real rate;
}
parameters {
  real mu;
  vector[2] b;
  real<lower=0> s;
}
model {
  mu ~ std_normal();
  b ~ normal(0, 2);
  s ~ lognormal(0, 1);
  y ~ normal(mu + b[1] * x + b[2], s * rate);
}
