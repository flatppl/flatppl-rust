functions {
  vector normal_lpdf_fn(vector z, real m_2) {
    return -0.5 * (z - m_2) .^ 2;
  }
  vector inv_fn(vector z) {
    return 1.0 ./ (1.0 + z);
  }
  vector mean_rng_fn(vector z) {
    return z + 1.0;
  }
}
data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_x;
  vector[N_x] x;
}
parameters {
  real m;
}
model {
  m ~ std_normal();
  y_obs ~ normal(normal_lpdf_fn(x, m) + inv_fn(x) + mean_rng_fn(x), 1.0);
}
