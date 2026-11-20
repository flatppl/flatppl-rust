data {
  real<lower=0, upper=1> xobs;
}
parameters {
  real<lower=0> alpha;
}
model {
  xobs ~ beta(alpha, 2.0);
}
