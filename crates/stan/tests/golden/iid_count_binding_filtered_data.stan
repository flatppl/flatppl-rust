functions {
  int lambda(real arg1) {
    return arg1 > 0.0;
  }
}
data {
  int<lower=0> N;
  vector[N] yd;
  int<lower=0> n;
}
transformed data {
  int yo_n = 0;
  for (i in 1:N) {
    yo_n += lambda(yd[i]);
  }
  vector[yo_n] yo;
  {
    int b_2 = 1;
    for (i in 1:N) {
      if (lambda(yd[i])) {
        yo[b_2] = yd[i];
        b_2 += 1;
      }
    }
  }
  if (yo_n != n) {
    reject("the size of yo is ", yo_n, ", but its law gives ", n);
  }
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  yo ~ normal(a, 1);
}
