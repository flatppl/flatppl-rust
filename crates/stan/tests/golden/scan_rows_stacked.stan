data {
  int<lower=0> K;
  real z;
  int<lower=0> T;
}
parameters {
  vector[K] w;
}
model {
  matrix[T, K] rows;
  for (i in 1:T) {
    rows[i] = (w * i + 1.0)';
  }

  w ~ std_normal();
  z ~ normal(sum(rows), 1);
}
