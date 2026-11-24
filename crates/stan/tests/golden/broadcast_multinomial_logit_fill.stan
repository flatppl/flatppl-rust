data {
  array[2, 3] int y;
}
parameters {
  vector[3] q;
}
model {
  for (i in 1:2) {
    if (sum(y[i]) != {10, 4}[i]) {
      target += negative_infinity();
    } else {
      y[i] ~ multinomial_logit(q);
    }
  }
}
