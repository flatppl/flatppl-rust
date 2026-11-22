functions {
  real flatppl_gamma_log_glm_lpdf(vector y, matrix x, real alpha, vector beta, real shape) {
    // Gamma(shape, shape * exp(-eta)) in link scale: log rate = log(shape) - eta. lmultiply(0, 0) = 0
    // gives the right-hand limit at y = 0 for shape = 1; shape < 1 and shape > 1 give +inf and -inf.
    if (min(y) < 0) return negative_infinity();
    vector[rows(x)] eta = alpha + x * beta;
    return rows(x) * (shape * log(shape) - lgamma(shape)) - shape * sum(eta)
           + sum(lmultiply(shape - 1, y)) - shape * dot_product(exp(-eta), y);
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  vector[N] y;
  matrix[N_x_1, N_x_2] x;
}
parameters {
  real alpha;
  vector[2] beta;
  real<lower=0> shape;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  shape ~ exponential(1);
  y ~ flatppl_gamma_log_glm(x, alpha, beta, shape);
}
