data {
  vector[4] y;
}
parameters {
  real d;
  real<lower=0> sv;
  vector[4] xv;
}
model {
  d ~ std_normal();
  sv ~ exponential(1.0);
  xv ~ normal(append_row(0.0, xv[1:3]) + d, sv);
  y ~ normal(xv, 0.5);
}
