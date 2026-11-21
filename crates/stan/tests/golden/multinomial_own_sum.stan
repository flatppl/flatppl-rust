data {
  int<lower=0> N;
  array[N] int cd;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 2.0, 3.0]');
  cd ~ multinomial(p);
}
