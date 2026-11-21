data {
  real y;
}
parameters {
  matrix[2, 2] M;
}
model {
  vector[2] scan_state = [0.1, 0.2]';
  for (t in 1:2) {
    scan_state = scan_state + M' * scan_state + (M')[({1, 2})[t], 1];
  }

  y ~ normal(sum(scan_state), 1.0);
}
