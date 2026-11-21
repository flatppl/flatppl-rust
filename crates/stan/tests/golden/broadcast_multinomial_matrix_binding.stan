data {
  int<lower=0> N;
  array[2, N] int y;
  matrix[2, N] P;
}
model {
  for (i in 1:2) {
    if (sum(y[i]) != {10, 4}[i]) {
      target += negative_infinity();
    } else {
      y[i] ~ multinomial(to_vector(P[i]));
    }
  }
}
