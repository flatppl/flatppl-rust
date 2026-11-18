functions {
  real flatppl_continued_poisson_lpdf(real x, real rate) {
    if (x < 0) return negative_infinity();
    return lmultiply(x, rate) - rate - lgamma(x + 1);
  }
  real flatppl_continued_poisson_lpdf(vector x, vector rate) {
    if (num_elements(x) > 0 && min(x) < 0) return negative_infinity();
    return sum(lmultiply(x, rate) - rate - lgamma(x + 1));
  }
  real flatppl_continued_poisson_lpdf(vector x, real rate) {
    if (num_elements(x) > 0 && min(x) < 0) return negative_infinity();
    return sum(lmultiply(x, rate)) - num_elements(x) * rate - sum(lgamma(x + 1));
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_x;
  int<lower=0> N_y_data;
  vector[N] w_data;
  vector[N_x] x;
  vector[N_y_data] y_data;
}
parameters {
  real<lower=0> a;
}
model {
  vector[N_x] mu = a * x;

  a ~ exponential(1);
  for (i in 1:N) {
    target += w_data[i] * flatppl_continued_poisson_lpdf(y_data[i] | mu[i]);
  }
}
