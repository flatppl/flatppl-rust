data {
  int<lower=0> N;
  int<lower=0> N_x;
  int<lower=0> N_y_data;
  vector[N] w_data;
  vector[N_x] x;
  array[N_y_data] int y_data;
}
parameters {
  real b;
  real<lower=0, upper=1> zi;
}
model {
  vector[N_x] mu = exp(b * x);

  b ~ std_normal();
  zi ~ beta(2, 2);
  for (i in 1:N) {
    if (y_data[i] == 0) {
      target += w_data[i] * log_mix(zi, 0, poisson_lpmf(0 | mu[i]));
    } else {
      target += w_data[i] * (log1m(zi) + poisson_lpmf(y_data[i] | mu[i]));
    }
  }
}
