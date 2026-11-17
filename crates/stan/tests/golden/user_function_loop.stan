functions {
  vector lagged(int t, vector y_obs, int K) {
    return y_obs[reverse(linspaced_int_array(K, t - K, t - 1))];
  }
}
data {
  int<lower=0> N;
  int<lower=0> K;
  vector[N] y_obs;
  int<lower=0> T;
}
transformed data {
  array[T - K] int times = linspaced_int_array(T - K, K + 1, T);
  vector[T - K] y = y_obs[times];
  matrix[T - K, K] X;
  for (i in 1:T - K) {
    X[i] = lagged(times[i], y_obs, K)';
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
