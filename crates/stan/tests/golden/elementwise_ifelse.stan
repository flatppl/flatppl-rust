data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
}
transformed data {
  vector[N] weights;
  for (i in 1:N) {
    weights[i] = x[i] > 1.5 ? 2.0 : 1.0;
  }
}
parameters {
  real lower_;
  real<lower=0> offset_;
}
model {
  lower_ ~ std_normal();
  offset_ ~ exponential(1.0);
  y_obs ~ normal(lower_ + x, offset_ * weights);
}
