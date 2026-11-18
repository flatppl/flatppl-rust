data {
  int<lower=0> I;
  array[I] int k_obs;
}
parameters {
  vector[I - 1] b_free;
}
model {
  vector[I] b = append_row(b_free, [-sum(b_free)]');

  target += normal_lpdf(append_row(b_free, [-sum(b_free)]') | 0, 3);
  k_obs ~ bernoulli_logit(-b);
}
