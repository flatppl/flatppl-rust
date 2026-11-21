data {
  int<lower=0> N;
  array[N] int c;
  int<lower=0> trials;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 1.0, 1.0]');
  if (sum(c) != trials) {
    target += negative_infinity();
  } else {
    c ~ multinomial(p);
  }
}
