data {
  array[5] int k_data;
}
transformed data {
  vector[5] k_data_lag = append_row(1, to_vector(k_data[1:4]));
}
parameters {
  real b;
}
model {
  b ~ std_normal();
  k_data ~ poisson_log(b * k_data_lag);
}
