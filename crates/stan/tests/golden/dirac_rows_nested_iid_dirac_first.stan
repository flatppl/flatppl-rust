data {
  int<lower=0> J;
  array[3, J] int counts;
}
transformed data {
  int counts_n_atom = 0;
  int counts_rest_n_atom = 0;
  int counts_n_other = 0;
  for (i in 1:3) {
    int r = 0;
    for (j in 1:J) {
      r += counts[i, j] == 0;
    }
    if (r == J) {
      counts_n_atom += 1;
    } else {
      counts_rest_n_atom += r;
    }
    counts_n_other += J - r;
  }
  array[counts_n_other] int counts_other;
  {
    int b = 1;
    for (i in 1:3) {
      for (j in 1:J) {
        if (counts[i, j] != 0) {
          counts_other[b] = counts[i, j];
          b += 1;
        }
      }
    }
  }
}
parameters {
  real<lower=0, upper=1> w;
  real<lower=0, upper=1> q;
  real<lower=0> lam;
}
model {
  w ~ beta(2, 2);
  q ~ beta(1, 3);
  lam ~ exponential(1);
  target += counts_n_atom * log_mix(w, 0, J * log_mix(q, 0, poisson_lpmf(0 | lam)));
  target += (3 - counts_n_atom) * log1m(w) + counts_rest_n_atom * log_mix(q, 0, poisson_lpmf(0 | lam)) + counts_n_other * log1m(q) + poisson_lpmf(counts_other | lam);
}
