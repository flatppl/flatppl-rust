data {
  int<lower=0> N;
  array[N] int n;
}
parameters {
  simplex[3] p;
}
model {
  p ~ dirichlet([1.0, 1.0, 1.0]');
  target += log_sum_exp([log(0.4) + (sum(n) != 10 ? negative_infinity() : multinomial_lpmf(n | p)), log(0.6) + (sum(n) != 10 ? negative_infinity() : multinomial_lpmf(n | [0.2, 0.3, 0.5]'))]');
}
