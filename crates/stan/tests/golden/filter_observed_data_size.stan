functions {
  vector shift(array[] int i) {
    return 0.5 * to_vector(i);
  }
  int keep(int i, vector y_all) {
    return y_all[i] > 0;
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_y_all;
  vector[N] y;
  vector[N_y_all] y_all;
}
transformed data {
  int idx_n = 0;
  for (i in 1:N_y_all) {
    idx_n += keep(i, y_all);
  }
  array[idx_n] int idx;
  {
    int b = 1;
    for (i in 1:N_y_all) {
      if (keep(i, y_all)) {
        idx[b] = i;
        b += 1;
      }
    }
  }
  vector[idx_n] means = shift(idx);
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y ~ normal(mu + means, 1);
}
