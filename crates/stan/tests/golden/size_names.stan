data {
  int<lower=0> J;
  vector[J] y_obs;
  vector[J] x;
}
parameters {
  vector[J] theta;
  real<lower=0> sigma;
}
model {
  theta ~ std_normal();
  sigma ~ exponential(1.0);
  y_obs ~ normal(theta .* x, sigma);
}
