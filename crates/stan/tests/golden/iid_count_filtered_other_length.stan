functions {
  int lambda(real arg1) {
    return arg1 > 0.0;
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_yd;
  vector[N] b;
  vector[N_yd] yd;
}
transformed data {
  int ys_n = 0;
  for (i in 1:N_yd) {
    ys_n += lambda(yd[i]);
  }
  vector[ys_n] ys;
  {
    int b_2 = 1;
    for (i in 1:N_yd) {
      if (lambda(yd[i])) {
        ys[b_2] = yd[i];
        b_2 += 1;
      }
    }
  }
  if (N != num_elements(ys)) {
    reject("the size of b is ", N, ", but its law gives ", num_elements(ys));
  }
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  b ~ normal(a, 1);
}
