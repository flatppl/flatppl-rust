functions {
  int keep(int i, vector y_all) {
    return y_all[i] > 0;
  }
  vector shift(array[] int i) {
    return 0.5 * to_vector(i);
  }
}
data {
  int<lower=0> N;
  vector[N] y_all;
}
transformed data {
  int idx_n = 0;
  for (i in 1:N) {
    idx_n += keep(i, y_all);
  }
  array[idx_n] int idx;
  {
    int b = 1;
    for (i in 1:N) {
      if (keep(i, y_all)) {
        idx[b] = i;
        b += 1;
      }
    }
  }
  vector[idx_n] ys = y_all[idx];
  vector[idx_n] means = shift(idx);
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  ys ~ normal(mu + means, 1);
}
