data {
  vector[1] x;
}
transformed data {
  vector[1] x_lag = append_row(0.5, x[1:0]);
}
parameters {
  real drift;
}
model {
  drift ~ std_normal();
  x ~ normal(x_lag + drift, 1.0);
}
