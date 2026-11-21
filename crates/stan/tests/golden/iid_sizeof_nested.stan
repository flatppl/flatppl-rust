data {
  int<lower=0> N;
  vector[N] X;
  array[2] vector[dims(X)[1]] b;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  for (i in 1:2) {
    b[i] ~ normal(a + mean(X), 1.0);
  }
}
