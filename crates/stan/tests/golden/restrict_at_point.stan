data {
  vector[5] obs_data;
}
parameters {
  real t1;
  real<lower=0> t2;
}
model {
  t1 ~ std_normal();
  t2 ~ gamma(2, 1);
  obs_data ~ normal(t1, t2);
}
