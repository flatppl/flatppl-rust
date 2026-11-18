data {
  int<lower=0> T;
  vector[T] y_obs;
  real s1;
}
transformed data {
  vector[T - 1] previous = y_obs[1:(T - 1)];
}
parameters {
  real mu;
  real<lower=0> a0;
  real<lower=0, upper=1> a1;
}
model {
  vector[T - 1] sigma_scan;
  {
    real sigma_state = s1;
    for (t in 1:(T - 1)) {
      sigma_state = sqrt(a0 + a1 * (previous[t] - mu) ^ 2 + 0.5 * sigma_state ^ 2);
      sigma_scan[t] = sigma_state;
    }
  }
  vector[T] sigma = append_row([s1]', sigma_scan);

  mu ~ std_normal();
  a0 ~ exponential(1);
  a1 ~ beta(1, 2);
  y_obs ~ normal(mu, sigma);
}
