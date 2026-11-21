data {
  vector[5] y_obs;
}
parameters {
  real<lower=0, upper=1> p;
  real mu;
  real<lower=0> shape;
}
model {
  p ~ beta(2, 2);
  mu ~ std_normal();
  shape ~ exponential(1);
  for (i in 1:5) {
    target += log_mix(p, normal_lpdf(y_obs[i] | mu, 1.0), (y_obs[i] < 0 ? negative_infinity() : gamma_lpdf(y_obs[i] | shape, 2.0)));
  }
}
