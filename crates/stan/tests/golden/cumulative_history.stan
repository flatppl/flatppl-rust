functions {
  array[] int history(int j_2, array[,] int shocks, array[] int previous) {
    return append_array({0}, cumulative_sum(shocks[j_2, previous]));
  }
  array[] int avoided(int j, int T, array[,] int shocks, array[] int previous) {
    array[T] int avoided_part = history(j, shocks, previous);
    array[T] int avoided_value;
    for (i in 1:T) {
      avoided_value[i] = i - 1 - avoided_part[i];
    }
    return avoided_value;
  }
}
data {
  int<lower=0> T;
  int<lower=0> J;
  array[J, T] int shocks;
}
transformed data {
  array[T - 1] int previous = linspaced_int_array(T - 1, 1, T - 1);
  array[J, T] int n_avoid;
  for (i in 1:J) {
    n_avoid[i] = avoided(i, T, shocks, previous);
  }
  array[J, T] int n_shock;
  for (i in 1:J) {
    n_shock[i] = history(i, shocks, previous);
  }
}
parameters {
  vector[2] beta;
}
model {
  beta ~ normal(0, 10);
  to_array_1d(shocks) ~ bernoulli_logit(to_vector((beta[1] * to_matrix(n_avoid) + beta[2] * to_matrix(n_shock))'));
}
