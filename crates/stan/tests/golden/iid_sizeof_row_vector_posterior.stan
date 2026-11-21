data {
  int<lower=0> N;
  vector[N] b;
}
transformed data {
  row_vector[3] X = [1.0, 2.0, 3.0];
  if (N != dims(X)[2]) {
    reject("the size of b is ", N, ", but its law gives ", dims(X)[2]);
  }
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1.0);
}
