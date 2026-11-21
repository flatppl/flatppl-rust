data {
  int<lower=0> N;
  vector[N] x;
  matrix[num_elements(x), num_elements(x)] R;
}
parameters {
  real<lower=0> eta;
}
model {
  eta ~ exponential(1);
  R ~ lkj_corr(eta);
}
