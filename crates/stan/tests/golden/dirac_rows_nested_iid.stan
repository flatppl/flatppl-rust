data {
  int<lower=0> J;
  array[4, J] int counts;
  int<lower=0> K;
}
transformed data {
  int counts_n_atom = 0;
  for (i in 1:4) {
    counts_n_atom += max(counts[i]) == 0 && min(counts[i]) == 0;
  }
  array[counts_n_atom] int counts_atom;
  array[4 - counts_n_atom] int counts_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:4) {
      if (max(counts[i]) == 0 && min(counts[i]) == 0) {
        counts_atom[a] = i;
        a += 1;
      } else {
        counts_rest[b] = i;
        b += 1;
      }
    }
  }
  array[4] int counts_row_n_atom;
  int counts_n_other = 0;
  for (i in 1:4) {
    counts_row_n_atom[i] = 0;
    for (j in 1:J) {
      counts_row_n_atom[i] += counts[i, j] == 0;
    }
    counts_n_other += J - counts_row_n_atom[i];
  }
  array[counts_n_other] int counts_other;
  array[4] int counts_other_start;
  {
    int b = 1;
    for (i in 1:4) {
      counts_other_start[i] = b;
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
  real<lower=0, upper=1> Omega;
  vector[4] u;
  vector[4] v;
}
model {
  vector[4] psi = inv_logit(u);
  vector[4] theta = inv_logit(v);

  Omega ~ beta(2, 2);
  u ~ std_normal();
  v ~ std_normal();
  for (j in 1:counts_n_atom) {
    int i = counts_atom[j];
    target += log_mix(Omega, J * log_mix(psi[i], binomial_lpmf(0 | K, theta[i]), 0), 0);
  }
  for (j in 1:(4 - counts_n_atom)) {
    int i = counts_rest[j];
    int m = J - counts_row_n_atom[i];
    target += log(Omega) + counts_row_n_atom[i] * log_mix(psi[i], binomial_lpmf(0 | K, theta[i]), 0) + m * log(psi[i]) + binomial_lpmf(counts_other[counts_other_start[i]:(counts_other_start[i] + m - 1)] | K, theta[i]);
  }
}
