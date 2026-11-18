functions {
  array[] int history(int i, array[,] int data_) {
    return data_[i, :];
  }
}
data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> T;
  array[N, T] int data_;
}
transformed data {
  array[M, T] int histories;
  for (i in 1:M) {
    histories[i] = history(i, data_);
  }
  array[M, T] int histories_previous;
  histories_previous[:, 1] = rep_array(0, M);
  histories_previous[:, 2:T] = histories[:, 1:(T - 1)];
  array[M, T] int histories_select;
  for (i in 1:M) {
    for (j in 1:T) {
      histories_select[i, j] = histories_previous[i, j] == 1 ? 2 : 1;
    }
  }
}
parameters {
  real<lower=0> a;
  real<lower=0> b;
  real s;
}
model {
  a ~ exponential(1);
  b ~ exponential(1);
  s ~ std_normal();
  to_array_1d(histories) ~ neg_binomial_2(([b, a]')[to_array_1d(histories_select)], exp(s));
}
