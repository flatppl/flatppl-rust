data {
  int<lower=0> N;
  array[N] int n;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 2.0, 3.0]');
  if (sum(n) != 10) {
    target += negative_infinity();
  } else {
    n ~ multinomial(p);
  }
}
