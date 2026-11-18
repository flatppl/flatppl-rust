data {
  array[5] int y;
}
transformed data {
  int y_n_atom = 0;
  for (i in 1:5) {
    y_n_atom += y[i] == -1;
  }
  array[y_n_atom] int y_atom;
  array[5 - y_n_atom] int y_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:5) {
      if (y[i] == -1) {
        y_atom[a] = i;
        a += 1;
      } else {
        y_rest[b] = i;
        b += 1;
      }
    }
  }
}
parameters {
  real<lower=0, upper=1> w;
  real<lower=0> lam;
}
model {
  w ~ beta(2, 2);
  lam ~ exponential(1);
  target += y_n_atom * log(1 - w);
  target += (5 - y_n_atom) * log(w) + poisson_lpmf(y[y_rest] | lam);
}
