data {
  vector[4] yobs;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ std_normal();
  sigma ~ exponential(1);
  yobs ~ normal(mu, sigma);
}
