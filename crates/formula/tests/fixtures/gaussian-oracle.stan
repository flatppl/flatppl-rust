data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> g;
}
parameters {
  real temp_Intercept;
  real b;
  real<lower=0> sd_g;
  vector[J] z_g;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu = temp_Intercept + (x - mean(x)) * b + sd_g * z_g[g];
}
model {
  target += normal_lpdf(temp_Intercept | 1, 3);
  target += normal_lpdf(b | 0, 2);
  target += normal_lpdf(sd_g | 0, 1) - normal_lccdf(0 | 0, 1);
  target += std_normal_lpdf(z_g);
  target += normal_lpdf(sigma | 0, 1) - normal_lccdf(0 | 0, 1);
  target += normal_lpdf(y | mu, sigma);
}
