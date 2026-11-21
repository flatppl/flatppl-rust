functions {
  int lambda(real arg1) {
    return arg1 > 5.0;
  }
}
transformed data {
  int kept_n = 0;
  for (i in 1:1) {
    kept_n += lambda(([3.0]')[i]);
  }
  vector[kept_n] kept;
  {
    int b_2 = 1;
    for (i in 1:1) {
      if (lambda(([3.0]')[i])) {
        kept[b_2] = ([3.0]')[i];
        b_2 += 1;
      }
    }
  }
  int x_n = 0;
  for (i in 1:2) {
    x_n += lambda(([1.0, 2.0]')[i]);
  }
  vector[x_n] x;
  {
    int b_2 = 1;
    for (i in 1:2) {
      if (lambda(([1.0, 2.0]')[i])) {
        x[b_2] = ([1.0, 2.0]')[i];
        b_2 += 1;
      }
    }
  }
  if (kept_n != num_elements(x)) {
    reject("the size of kept is ", kept_n, ", but its law gives ", num_elements(x));
  }
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  kept ~ normal(a, 1);
}
