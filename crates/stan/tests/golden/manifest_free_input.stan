data {
  int a;
}
parameters {
  real<lower=0> rate;
}
model {
  a ~ poisson(rate);
}
