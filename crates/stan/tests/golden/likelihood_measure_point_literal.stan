data {
  vector[2] y;
}
transformed data {
  real mu = 0.3;
  real sg = 2.0;
}
model {
  y ~ normal(mu, sg);
}
