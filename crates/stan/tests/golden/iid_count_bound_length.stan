data {
  int<lower=0> N;
  vector[N] y_data;
}
transformed data {
  int n = num_elements(y_data);
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y_data ~ normal(mu, 1.0);
}
