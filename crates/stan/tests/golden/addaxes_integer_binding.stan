data {
  int<lower=0> N;
  int<lower=0> N_k;
  real y;
  vector[N] x_obs;
  array[N_k] int k;
  int n;
}
transformed data {
  array[N_k, 1] int kk;
  for (i in 1:N_k) {
    for (j in 1:1) {
      kk[i, j] = k[i];
    }
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + x_obs[kk[2, 1]], 1);
  n ~ binomial(kk[1, 1] + 4, 0.3);
}
