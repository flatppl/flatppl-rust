data {
  int<lower=0> N;
  int<lower=0> N_S_1;
  int<lower=0> N_S_2;
  array[2] vector[N] y;
  matrix[N_S_1, N_S_2] S;
}
transformed data {
  vector[2] x = [1.0, 2.0]';
}
parameters {
  real mu;
  real nu;
}
model {
  for (i in 1:2) {
    y[i] ~ multi_normal([x[i] - (mu - nu), 2.0 / (mu - nu)]', S);
  }
}
