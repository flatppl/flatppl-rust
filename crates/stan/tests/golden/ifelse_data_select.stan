data {
  int<lower=0> N;
  vector[N] y_obs;
  array[N] int flag;
}
transformed data {
  array[N] int select;
  for (i in 1:N) {
    select[i] = flag[i] == 1 ? 2 : 1;
  }
}
parameters {
  real a;
  real b;
}
model {
  a ~ std_normal();
  b ~ std_normal();
  y_obs ~ normal(([b, a]')[select], 1);
}
