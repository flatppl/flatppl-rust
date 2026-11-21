data {
  int<lower=0> N;
  int<lower=0> N_x;
  vector[N] y;
  vector[N_x] x;
}
transformed data {
  int n = num_elements(x);
  if (N != n) {
    reject("the size of y is ", N, ", but its law gives ", n);
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu, 1.0);
}
