data {
  vector[4] y_obs;
  vector[4] x;
}
transformed data {
  vector[4] weights;
  for (i in 1:4) {
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
