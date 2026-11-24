data {
  int<lower=0> J;
  vector[8] ys;
  vector[8] sig;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector<offset=rep_vector(mu, J), multiplier=tau>[8] theta;
}
model {
  theta ~ normal(rep_vector(mu, J), tau);
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5) T[0, ];
  ys ~ normal(theta, sig);
}
