data {
  int<lower=0> N;
  int<lower=0> N_x;
  int<lower=0> N_y_data;
  vector[N] w_data;
  vector[N_x] x;
  vector[N_y_data] y_data;
}
parameters {
  real a;
  real<lower=0> sigma;
}
model {
  vector[N_x] mu = a + x;

  a ~ normal(0, 2);
  sigma ~ exponential(1);
  for (i in 1:N) {
    target += w_data[i] * normal_lpdf(y_data[i] | mu[i], sigma);
  }
}
