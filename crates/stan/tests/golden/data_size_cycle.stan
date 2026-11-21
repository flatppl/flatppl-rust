data {
  int<lower=0> N;
  vector[N] yd;
  vector[num_elements(yd)] xd;
}
transformed data {
  if (N != num_elements(xd)) {
    reject("the size of yd is ", N, ", but its law gives ", num_elements(xd));
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  xd ~ normal(mu, 1.0);
  yd ~ normal(mu, 1.0);
}
