functions {
  real f(real acc, real x) {
    return 0.5 * acc + x;
  }
}
data {
  real y;
}
parameters {
  real a0;
  real xv;
}
model {
  a0 ~ std_normal();
  xv ~ std_normal();
  y ~ normal(f(3.0, 1.0), 1);
}
