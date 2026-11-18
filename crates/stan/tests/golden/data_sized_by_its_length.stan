data {
  int<lower=0> N;
  array[N] int counts_data;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([2.0, 3.0, 4.0]');
  counts_data ~ categorical(p);
}
