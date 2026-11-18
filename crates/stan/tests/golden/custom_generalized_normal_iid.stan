functions {
  real flatppl_generalized_normal_lpdf(real x, real mu, real alpha, real beta) {
    return log(beta) - log2() - log(alpha) - lgamma(1 / beta) - pow(abs(x - mu) / alpha, beta);
  }
  real flatppl_generalized_normal_lpdf(vector x, real mu, real alpha, real beta) {
    return num_elements(x) * (log(beta) - log2() - log(alpha) - lgamma(1 / beta))
           - sum((abs(x - mu) / alpha) .^ beta);
  }
}
data {
  vector[5] y_obs;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y_obs ~ flatppl_generalized_normal(mu, 1.3, 1.5);
}
