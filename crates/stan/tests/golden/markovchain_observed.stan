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
  array[T] int histories_atom_select;
  histories_atom_select[1] = (0) == 1 ? 2 : 1;
  histories_atom_select[2:T] = rep_array((0) == 1 ? 2 : 1, T - 1);
  int histories_n_atom = 0;
  for (i in 1:M) {
    histories_n_atom += max(histories[i]) == 0 && min(histories[i]) == 0;
  }
  array[M - histories_n_atom] int histories_rest;
  {
    int b = 1;
    for (i in 1:M) {
      if (!(max(histories[i]) == 0 && min(histories[i]) == 0)) {
        histories_rest[b] = i;
        b += 1;
      }
    }
  }
}
parameters {
  real<lower=0, upper=1> omega;
  real<lower=0, upper=1> p;
  real<lower=0, upper=1> c;
}
model {
  omega ~ uniform(0, 1);
  p ~ uniform(0, 1);
  c ~ uniform(0, 1);
  target += histories_n_atom * log_mix(omega, bernoulli_lpmf(rep_array(0, T) | ([p, c]')[histories_atom_select]), 0);
  target += (M - histories_n_atom) * log(omega) + bernoulli_lpmf(to_array_1d(histories[histories_rest]) | ([p, c]')[to_array_1d(histories_select[histories_rest])]);
}
