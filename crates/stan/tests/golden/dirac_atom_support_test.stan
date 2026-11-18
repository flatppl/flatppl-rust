data {
  array[5] int y;
  int<lower=0> N;
}
transformed data {
  int y_n_atom = 0;
  for (i in 1:5) {
    y_n_atom += y[i] == 3;
  }
  array[y_n_atom] int y_atom;
  array[5 - y_n_atom] int y_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:5) {
      if (y[i] == 3) {
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
  real<lower=0, upper=1> p;
}
model {
  w ~ beta(2, 2);
  p ~ beta(1, 1);
  target += y_n_atom * log_mix(w, ((3 > N) ? negative_infinity() : binomial_lpmf(3 | N, p)), 0);
  target += (5 - y_n_atom) * log(w) + binomial_lpmf(y[y_rest] | N, p);
}
