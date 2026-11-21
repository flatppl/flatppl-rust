data {
  int<lower=0> N;
  int<lower=1> n;
  array[N] int a;
}
parameters {
  simplex[3] p;
}
model {
  if (sum(a) != n) {
    target += negative_infinity();
  } else {
    a ~ multinomial(p);
  }
}
