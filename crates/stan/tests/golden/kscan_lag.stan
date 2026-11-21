data {
  int<lower=0> N;
  vector[N] traj_obs;
  vector[N] dts;
}
transformed data {
  vector[N] traj_obs_lag = append_row(0.0, traj_obs[1:(N - 1)]);
}
parameters {
  real<lower=0> D;
}
model {
  D ~ gamma(2.0, 0.5);
  traj_obs ~ normal(traj_obs_lag, sqrt(2.0 * D * dts));
}
