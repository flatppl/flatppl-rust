data {
  vector[4] x_data;
}
transformed data {
  vector[4] x_data_lag = append_row(0.3, x_data[1:3]);
}
parameters {
  vector[2] w;
}
model {
  w ~ std_normal();
  x_data ~ normal(w[1] * x_data_lag + w[2], 0.7);
}
