data {
  int<lower=0> N;
  vector[N] X;
  vector[dims(X)[1]] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1.0);
}
