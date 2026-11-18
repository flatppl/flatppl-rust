data {
  int<lower=0> N;
  int<lower=0> N_z;
  array[N] int y;
  array[N] int trials;
  vector[N_z] z;
  array[N_z] int x;
}
transformed data {
  array[N] int n;
  for (i in 1:N) {
    n[i] = max(trials[i], 1);
  }
  vector[N_z] ind = ([1.0, 0.0, 0.5]')[x];
}
parameters {
  real<lower=0, upper=1> p;
  real b;
}
model {
  p ~ beta(2, 2);
  b ~ std_normal();
  y ~ binomial(n, p);
  z ~ normal(b * ind, 1);
}
