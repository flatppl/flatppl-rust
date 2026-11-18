data {
  vector[3] y;
  vector[3] x;
}
parameters {
  vector[3] b;
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + b .* x, 1);
}
