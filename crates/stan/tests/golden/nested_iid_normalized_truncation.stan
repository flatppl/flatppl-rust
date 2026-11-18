data {
  array[2] vector[2] y;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  for (i in 1:2) {
    y[i] ~ normal(mu, 1) T[0, ];
  }
}
