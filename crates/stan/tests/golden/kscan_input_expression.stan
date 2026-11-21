data {
  vector[3] y;
}
transformed data {
  vector[3] y_lag = append_row(0.5, y[1:2]);
}
parameters {
  real mu;
}
model {
  y ~ normal(y_lag + 2.0 * ([1.0, 2.0, 3.0]' - mu), 1.0);
}
