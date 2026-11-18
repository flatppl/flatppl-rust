functions {
  vector f(vector acc, real x, real c) {
    vector[2] m1 = acc * c;
    vector[2] m2 = m1 + x;
    return m2;
  }
}
data {
  vector[2] y;
  real c;
}
parameters {
  vector[3] a0;
  real xv;
}
model {
  a0 ~ std_normal();
  xv ~ std_normal();
  y ~ normal(f([1.0, 2.0]', 0.5, c), 1);
}
