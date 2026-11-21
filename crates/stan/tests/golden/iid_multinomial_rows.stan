data {
  int<lower=0> N;
  array[2, N] int c;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 2.0, 3.0]');
  for (i in 1:2) {
    if (sum(c[i]) != 10) {
      target += negative_infinity();
    } else {
      c[i] ~ multinomial(p);
    }
  }
}
