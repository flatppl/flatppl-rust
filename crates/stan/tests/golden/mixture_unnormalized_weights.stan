data {
  int<lower=0> N;
  array[N] int k_obs;
}
transformed data {
  array[10] int sizes = linspaced_int_array(10, 1, 10);
  vector[10] weights = ones_vector(10);
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  {
    vector[10] lps;
    for (i in 1:10) {
      lps[i] = (max(k_obs) > sizes[i] ? negative_infinity() : binomial_lpmf(k_obs | sizes[i], theta));
    }
    target += log_sum_exp(log(weights) + lps) - log(sum(weights));
  }
}
