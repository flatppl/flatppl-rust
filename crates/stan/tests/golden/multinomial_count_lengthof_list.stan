data {
  int<lower=0> N;
  int<lower=0> N_x_1;
  int<lower=0> N_x_2;
  array[N] int c;
  array[N_x_1] vector[N_x_2] x;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 2.0, 3.0]');
  if (sum(c) != size(x)) {
    target += negative_infinity();
  } else {
    c ~ multinomial(p);
  }
}
