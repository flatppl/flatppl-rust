parameters {
  real eta;
}
transformed parameters {
  vector[2] y = [exp(eta), 0.5]';
}
model {
  y ~ beta(8.0, 8.0);
}
