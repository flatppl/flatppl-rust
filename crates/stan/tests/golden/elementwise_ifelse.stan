data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_x;
  vector[N_x] x;
}
transformed data {
  vector[N_x] weights;
  for (i in 1:N_x) {
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
