data {
  int<lower=0> M_obs;
  int<lower=0> C_obs;
  int<lower=0> R_obs;
}
transformed data {
  real rcp_max = 1.0 * M_obs / (C_obs - R_obs + M_obs);
}
parameters {
  real<lower=0, upper=rcp_max> rcp;
}
model {
  rcp ~ uniform(0, rcp_max);
  R_obs ~ binomial(C_obs, rcp);
}
