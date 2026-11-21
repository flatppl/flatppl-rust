data {
  vector[4] x_data;
}
transformed data {
  vector[4] x_data_lag = append_row(0.2, x_data[1:3]);
}
parameters {
  real a;
  real<lower=0> s;
}
model {
  a ~ std_normal();
  s ~ exponential(1.0);
  x_data ~ normal(a * x_data_lag + 0.5, s * (1.0 + abs(x_data_lag)));
}
