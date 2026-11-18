data {
  real z;
}
parameters {
  array[2] vector[3] w;
}
model {
  for (i in 1:2) {
    w[i] ~ normal(0, 5);
  }
  z ~ normal(sum(w[1]) - w[2][3], 1);
}
