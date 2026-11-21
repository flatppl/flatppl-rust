data {
  int<lower=0> N;
  vector[N] b;
}
transformed data {
  int n = 2 + 1;
  if (N != n) {
    reject("the size of b is ", N, ", but its law gives ", n);
  }
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1);
}
