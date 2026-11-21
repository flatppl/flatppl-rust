data {
  real y;
}
parameters {
  vector[3] q;
}
model {
  array[2] vector[3] M = rep_array(q, 2);

  y ~ normal(M[1, 3], 1.0);
}
