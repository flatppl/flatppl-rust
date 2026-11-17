functions {
  vector lagged(int t, vector y_obs) {
    return y_obs[reverse(linspaced_int_array(2, t - 2, t - 1))];
  }
}
data {
  int<lower=0> K;
  int<lower=0> T;
  vector[T] y_obs;
}
transformed data {
  array[6] int times = linspaced_int_array(6, 3, 8);
  vector[6] y = y_obs[times];
  matrix[6, K] X;
  for (i in 1:6) {
    X[i] = lagged(times[i], y_obs)';
  }
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ std_normal();
  beta ~ std_normal();
  sigma ~ exponential(1.0);
  y ~ normal(alpha + X * beta, sigma);
}
