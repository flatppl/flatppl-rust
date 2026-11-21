data {
  int<lower=0> N;
  array[2, N] int y;
}
transformed data {
  array[2] int arg1 = {10, 4};
}
model {
  for (i in 1:2) {
    if (sum(y[i]) != arg1[i]) {
      target += negative_infinity();
    } else {
      y[i] ~ multinomial([0.2, 0.3, 0.5]');
    }
  }
}
