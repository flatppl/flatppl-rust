data {
  array[2] vector[2] y;
}
parameters {
  vector<lower=0>[2] q;
}
model {
  for (i in 1:2) {
    y[i] ~ dirichlet(q);
  }
}
