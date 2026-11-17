data {
  int<lower=0> J;
  int<lower=0> I;
  array[I, J] int y_obs;
}
parameters {
  real<lower=0> sigma;
  vector[J] theta;
  vector<lower=0>[I] a;
  vector[I] b;
}
model {
  matrix[I, J] log_odds = rep_matrix(a, J) .* (rep_matrix(theta', I) - rep_matrix(b, J));

  sigma ~ exponential(1.0);
  theta ~ normal(0, sigma);
  a ~ lognormal(0, 1);
  b ~ std_normal();
  to_array_1d(y_obs) ~ bernoulli_logit(to_vector(log_odds'));
}
