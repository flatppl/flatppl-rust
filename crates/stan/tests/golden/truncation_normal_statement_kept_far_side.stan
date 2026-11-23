data {
  real y;
}
model {
  y ~ normal(1.0, 0.1) T[0.0, ];
}
