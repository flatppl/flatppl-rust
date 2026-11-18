functions {
  int keep(real x) {
    return x > 0;
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_i;
  real z;
  vector[N] xs;
  vector[N_i] i;
}
transformed data {
  int k_n = 0;
  for (j in 1:N) {
    k_n += keep(xs[j]);
  }
  vector[k_n] k;
  {
    int b = 1;
    for (j in 1:N) {
      if (keep(xs[j])) {
        k[b] = xs[j];
        b += 1;
      }
    }
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  z ~ normal(mu + sum(k) + sum(i), 1);
}
