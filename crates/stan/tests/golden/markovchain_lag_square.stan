data {
  vector[4] x_data;
}
transformed data {
  vector[4] x_data_lag = append_row(0.3, x_data[1:3]);
}
parameters {
  real d;
}
model {
  d ~ std_normal();
  x_data ~ normal(d * x_data_lag, 0.5 + x_data_lag .* x_data_lag ./ (1.0 + x_data_lag .* x_data_lag));
}
