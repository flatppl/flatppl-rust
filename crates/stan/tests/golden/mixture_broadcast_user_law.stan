data {
  array[5] int counts;
  int<lower=0> T;
}
transformed data {
  int counts_n_atom = 0;
  for (i in 1:5) {
    counts_n_atom += counts[i] == 0;
  }
  array[counts_n_atom] int counts_atom;
  array[5 - counts_n_atom] int counts_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:5) {
      if (counts[i] == 0) {
        counts_atom[a] = i;
        a += 1;
      } else {
        counts_rest[b] = i;
        b += 1;
      }
    }
  }
}
parameters {
  real<lower=0, upper=1> omega;
  vector<lower=0, upper=1>[5] p;
}
model {
  omega ~ beta(1, 1);
  p ~ beta(1, 1);
  for (i in 1:counts_n_atom) {
    target += log_mix(omega, binomial_lpmf(0 | T, p[counts_atom[i]]), 0);
  }
  target += (5 - counts_n_atom) * log(omega) + binomial_lpmf(counts[counts_rest] | T, p[counts_rest]);
}
