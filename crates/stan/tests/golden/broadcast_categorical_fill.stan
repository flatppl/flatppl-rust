data {
  array[2] int y;
}
parameters {
  simplex[3] q;
}
model {
  for (i in 1:2) {
    y[i] ~ categorical(q);
  }
}
