data {
  int<lower=0> N;
  vector[N] y_data;
  vector[N] x_data;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, sigma * 3);
  beta ~ normal(0, square(sigma));
  square(sigma) ~ inv_gamma(5, 5);
  target += log(2) + log(sigma);
  y_data ~ normal(alpha + beta * x_data, sigma);
}
