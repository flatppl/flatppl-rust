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
  vector[3] y;
}
parameters {
  real<lower=0> r;
}
model {
  target += flatppl_continued_poisson_lpdf(y | r);
}
