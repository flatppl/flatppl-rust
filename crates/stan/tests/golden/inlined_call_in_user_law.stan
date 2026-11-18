data {
  int<lower=0> N;
  vector[2] y;
  vector[N] vs;
}
parameters {
  vector[2] mus;
}
model {
  mus ~ std_normal();
  for (i in 1:2) {
    y[i] ~ normal(sum(vs * mus[i]), 1);
  }
}
