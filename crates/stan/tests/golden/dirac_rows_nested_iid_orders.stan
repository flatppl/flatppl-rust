data {
  int<lower=0> J;
  array[3, J] int counts1;
  array[3, J] int counts2;
}
transformed data {
  int counts1_n_atom = 0;
  int counts1_rest_n_atom = 0;
  int counts1_n_other = 0;
  for (i in 1:3) {
    int r = 0;
    for (j in 1:J) {
      r += counts1[i, j] == 0;
    }
    if (r == J) {
      counts1_n_atom += 1;
    } else {
      counts1_rest_n_atom += r;
    }
    counts1_n_other += J - r;
  }
  array[counts1_n_other] int counts1_other;
  {
    int b = 1;
    for (i in 1:3) {
      for (j in 1:J) {
        if (counts1[i, j] != 0) {
          counts1_other[b] = counts1[i, j];
          b += 1;
        }
      }
    }
  }
  int counts2_n_atom = 0;
  for (i in 1:3) {
    counts2_n_atom += max(counts2[i]) == 0 && min(counts2[i]) == 0;
  }
  array[counts2_n_atom] int counts2_atom;
  array[3 - counts2_n_atom] int counts2_rest;
  {
    int a = 1;
    int b = 1;
    for (i in 1:3) {
      if (max(counts2[i]) == 0 && min(counts2[i]) == 0) {
        counts2_atom[a] = i;
        a += 1;
      } else {
        counts2_rest[b] = i;
        b += 1;
      }
    }
  }
  array[3] int counts2_row_n_atom;
  int counts2_n_other = 0;
  for (i in 1:3) {
    counts2_row_n_atom[i] = 0;
    for (j in 1:J) {
      counts2_row_n_atom[i] += counts2[i, j] == 0;
    }
    counts2_n_other += J - counts2_row_n_atom[i];
  }
  array[counts2_n_other] int counts2_other;
  array[3] int counts2_other_start;
  {
    int b = 1;
    for (i in 1:3) {
      counts2_other_start[i] = b;
      for (j in 1:J) {
        if (counts2[i, j] != 0) {
          counts2_other[b] = counts2[i, j];
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
  vector[3] u;
}
model {
  vector[3] p = inv_logit(u);

  w ~ beta(2, 2);
  q ~ beta(1, 3);
  lam ~ exponential(1);
  u ~ std_normal();
  target += counts1_n_atom * log_mix(w, J * log_mix(q, poisson_lpmf(0 | lam), 0), 0);
  target += (3 - counts1_n_atom) * log(w) + counts1_rest_n_atom * log_mix(q, poisson_lpmf(0 | lam), 0) + counts1_n_other * log(q) + poisson_lpmf(counts1_other | lam);
  for (j in 1:counts2_n_atom) {
    int i = counts2_atom[j];
    target += log_mix(w, 0, J * log_mix(p[i], 0, poisson_lpmf(0 | lam)));
  }
  for (j in 1:(3 - counts2_n_atom)) {
    int i = counts2_rest[j];
    int m = J - counts2_row_n_atom[i];
    target += log1m(w) + counts2_row_n_atom[i] * log_mix(p[i], 0, poisson_lpmf(0 | lam)) + m * log1m(p[i]) + poisson_lpmf(counts2_other[counts2_other_start[i]:(counts2_other_start[i] + m - 1)] | lam);
  }
}
