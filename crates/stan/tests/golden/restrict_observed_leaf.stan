data {
  vector[5] obs_data;
}
parameters {
  real theta1;
  real<lower=0> theta2;
}
model {
  theta1 ~ std_normal();
  theta2 ~ gamma(2, 1);
  obs_data ~ normal(theta1, theta2);
}
