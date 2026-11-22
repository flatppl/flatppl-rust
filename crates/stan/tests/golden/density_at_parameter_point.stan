parameters {
  real eta;
}
model {
  vector[2] y = [exp(eta), 0.5]';

  y ~ beta(8.0, 8.0);
}
