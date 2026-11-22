data {
  int<lower=0> N;
  array[N] int c;
  int<lower=0> n;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 2.0, 3.0]');
  if (sum(c) != n) {
    target += negative_infinity();
  } else {
    c ~ multinomial(p);
  }
}
