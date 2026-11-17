functions {
  vector growth(vector t, real a_2) {
    return exp(a_2 * t) + t;
  }
}
data {
  vector[4] y_obs;
  vector[4] x;
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
