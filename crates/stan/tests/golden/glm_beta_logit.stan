functions {
  real flatppl_beta_logit_glm_lpdf(vector y, matrix x, real alpha, vector beta, real precision) {
    // Beta(precision * invlogit(eta), precision * invlogit(-eta)): each shape keeps its digits in its own
    // tail. lmultiply(0, 0) = 0 gives the limit at y = 0 or 1 for a shape of 1; a shape below or above 1
    // gives +inf or -inf there.
    if (min(y) < 0 || max(y) > 1) return negative_infinity();
    vector[rows(x)] eta = alpha + x * beta;
    vector[rows(x)] a = precision * inv_logit(eta);
    vector[rows(x)] b = precision * inv_logit(-eta);
    return sum(lmultiply(a - 1, y) + lmultiply(b - 1, 1 - y) - lbeta(a, b));
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
  real<lower=0> precision;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  precision ~ exponential(0.2);
  y ~ flatppl_beta_logit_glm(x, alpha, beta, precision);
}
