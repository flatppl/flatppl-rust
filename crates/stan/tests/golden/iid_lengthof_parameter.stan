data {
  vector[3] b;
}
parameters {
  vector[3] X;
}
model {
  X ~ std_normal();
  b ~ normal(X[1], 1.0);
}
