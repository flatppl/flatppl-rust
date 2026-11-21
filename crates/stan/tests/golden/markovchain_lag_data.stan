data {
  vector[5] x_data;
}
transformed data {
  vector[5] x_data_lag = append_row(0.0, x_data[1:4]);
}
parameters {
  real drift;
  real<lower=0> s;
}
model {
  drift ~ std_normal();
  s ~ cauchy(0, 1) T[0, ];
  x_data ~ normal(x_data_lag + drift, s);
}
