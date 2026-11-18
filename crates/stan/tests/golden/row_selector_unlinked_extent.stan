functions {
  array[] int history(int i, array[,] int data_) {
    return data_[i, :];
  }
}
data {
  int<lower=0> N;
  int<lower=0> T;
  int<lower=0> M;
  array[N, T] int data_;
}
transformed data {
  array[M, T] int histories;
  for (i in 1:M) {
    histories[i] = history(i, data_);
  }
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
  vector<lower=0, upper=1>[T] p;
}
model {
  omega ~ uniform(0, 1);
  p ~ uniform(0, 1);
  target += histories_n_atom * log_mix(omega, bernoulli_lpmf(rep_array(0, T) | p), 0);
  target += (M - histories_n_atom) * log(omega) + bernoulli_lpmf(to_array_1d(histories[histories_rest]) | to_vector(rep_matrix(p, M - histories_n_atom)));
}
