functions {
  vector growth(vector t, real a_2) {
    return exp(a_2 * t) + t;
  }
}
data {
  int<lower=0> N_y_obs;
  vector[N_y_obs] y_obs;
  int<lower=0> N_x;
  vector[N_x] x;
}
parameters {
  real a;
  real k;
}
model {
  a ~ std_normal();
  k ~ std_normal();
  y_obs ~ normal(2.0 * growth(x, a) - k, 0.5);
}
