functions {
  int lambda(real arg1) {
    return arg1 > 0;
  }
  int lambda_2(real arg1) {
    return arg1 < 2;
  }
  int lambda_3(real arg1) {
    return arg1 < 2.2;
  }
  int lambda_4(real arg1) {
    return arg1 > 0.4;
  }
}
data {
  int<lower=0> N;
  real z;
  vector[N] y_all;
  real w;
}
transformed data {
  int a_n = 0;
  for (i in 1:N) {
    a_n += lambda(y_all[i]);
  }
  vector[a_n] a;
  {
    int b_2 = 1;
    for (i in 1:N) {
      if (lambda(y_all[i])) {
        a[b_2] = y_all[i];
        b_2 += 1;
      }
    }
  }
  int b_n = 0;
  for (i in 1:a_n) {
    b_n += lambda_2(a[i]);
  }
  vector[b_n] b;
  {
    int b_2 = 1;
    for (i in 1:a_n) {
      if (lambda_2(a[i])) {
        b[b_2] = a[i];
        b_2 += 1;
      }
    }
  }
  int kept_n = 0;
  for (i in 1:N) {
    kept_n += lambda_3(y_all[i]);
  }
  vector[kept_n] kept;
  {
    int b_2 = 1;
    for (i in 1:N) {
      if (lambda_3(y_all[i])) {
        kept[b_2] = y_all[i];
        b_2 += 1;
      }
    }
  }
  int c_n = 0;
  for (i in 1:kept_n) {
    c_n += lambda_4(kept[i]);
  }
  vector[c_n] c;
  {
    int b_2 = 1;
    for (i in 1:kept_n) {
      if (lambda_4(kept[i])) {
        c[b_2] = kept[i];
        b_2 += 1;
      }
    }
  }
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  z ~ normal(mu + sum(b), 1);
  w ~ normal(mu * sum(c), 1);
}
