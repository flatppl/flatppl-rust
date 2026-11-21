data {
  vector[4] y;
}
parameters {
  real drift;
  real x0;
  vector[4] x;
}
model {
  drift ~ std_normal();
  x0 ~ std_normal();
  x ~ normal(append_row(x0, x[1:3]) + drift, 0.5);
  y ~ normal(x, 0.3);
}
