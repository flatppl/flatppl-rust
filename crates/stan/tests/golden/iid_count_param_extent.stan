data {
  vector[3] y;
}
parameters {
  vector[3] v;
  real mu;
}
model {
  y ~ normal(mu, 1.0);
  v ~ std_normal();
}
