data {
  vector[3] obs;
}
parameters {
  real t1;
  real<lower=0> t2;
}
model {
  t1 ~ std_normal();
  t2 ~ exponential(1);
  obs ~ normal(t1, t2);
}
