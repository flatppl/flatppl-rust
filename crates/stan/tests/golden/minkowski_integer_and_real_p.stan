data {
  real y;
  real y_2;
}
parameters {
  vector[3] u;
  vector[3] v;
  vector[3] w;
}
model {
  y ~ normal(pow(sum(pow(abs(u - (v - w)), 3)), 1.0 / 3), 1.0);
  y_2 ~ normal(pow(sum(pow(abs(u - w), 1.5)), 1.0 / 1.5), 1.0);
}
