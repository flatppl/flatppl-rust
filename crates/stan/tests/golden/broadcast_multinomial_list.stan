data {
  array[2, 3] int y;
}
model {
  for (i in 1:2) {
    if (sum(y[i]) != {10, 4}[i]) {
      target += negative_infinity();
    } else {
      y[i] ~ multinomial({[0.2, 0.3, 0.5]', [0.1, 0.6, 0.3]'}[i]);
    }
  }
}
