functions {
  real flatppl_relativistic_breit_wigner_lpdf(real x, real m, real w) {
    if (x <= 0) return negative_infinity();
    real g = sqrt(square(m) * (square(m) + square(w)));
    real log_norm = log(pi()) + 0.5 * log(square(m) + g) - log(2 * sqrt2()) - log(m) - log(w) - log(g);
    return -log(square(square(x) - square(m)) + square(m * w)) - log_norm;
  }
  real flatppl_relativistic_breit_wigner_lpdf(vector x, real m, real w) {
    if (num_elements(x) > 0 && min(x) <= 0) return negative_infinity();
    real g = sqrt(square(m) * (square(m) + square(w)));
    real log_norm = log(pi()) + 0.5 * log(square(m) + g) - log(2 * sqrt2()) - log(m) - log(w) - log(g);
    return -sum(log(square(square(x) - square(m)) + square(m * w))) - num_elements(x) * log_norm;
  }
}
data {
  vector[5] y_obs;
}
parameters {
  real<lower=0> w;
}
model {
  w ~ exponential(0.5);
  y_obs ~ flatppl_relativistic_breit_wigner(91.19, w);
}
