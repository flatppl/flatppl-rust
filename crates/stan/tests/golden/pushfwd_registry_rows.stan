data {
  real z;
}
parameters {
  real<lower=0> v_exp;
  real v_log;
  real v_log10;
  real<lower=0> v_log1p;
  real<lower=-1> v_expm1;
  real v_logit;
  real<lower=0, upper=1> v_invlogit;
  real v_probit;
  real<lower=0, upper=1> v_invprobit;
  real<lower=-pi() / 2, upper=pi() / 2> v_atan;
  real v_sinh;
  real v_asinh;
  real<lower=-1, upper=1> v_tanh;
  real v_pow3;
  real v_pow5;
  real<lower=0> v_pow1p5;
  real<lower=0> v_powmhalf;
}
model {
  log(v_exp) ~ normal(0.3, 0.8);
  target += -log(v_exp);
  exp(v_log) ~ gamma(3, 2);
  target += v_log;
  pow(10, v_log10) ~ gamma(3, 2);
  target += v_log10 * log(10) + log(log(10));
  expm1(v_log1p) ~ exponential(1.5);
  target += v_log1p;
  log1p(v_expm1) ~ normal(0.2, 0.7);
  target += -log1p(v_expm1);
  inv_logit(v_logit) ~ beta(2, 3);
  target += log_inv_logit(v_logit) + log1m_inv_logit(v_logit);
  logit(v_invlogit) ~ normal(0.1, 1.2);
  target += -log(v_invlogit) - log1m(v_invlogit);
  Phi(v_probit) ~ beta(2, 3);
  target += std_normal_lpdf(v_probit);
  inv_Phi(v_invprobit) ~ normal(0.1, 0.9);
  target += -std_normal_lpdf(inv_Phi(v_invprobit));
  tan(v_atan) ~ normal(0.2, 1.5);
  target += -2 * log(cos(v_atan));
  asinh(v_sinh) ~ normal(0.1, 0.8);
  target += -0.5 * log1p(square(v_sinh));
  sinh(v_asinh) ~ normal(0.4, 2.0);
  target += abs(v_asinh) + log1p_exp(-2 * abs(v_asinh)) - log(2);
  atanh(v_tanh) ~ normal(0.1, 0.6);
  target += -log1m(square(v_tanh));
  cbrt(v_pow3) ~ normal(0.2, 1.0);
  target += -log(3) - 0.6666666666666667 * log(abs(v_pow3));
  (v_pow5 < 0 ? -pow(-v_pow5, 0.2) : pow(v_pow5, 0.2)) ~ normal(0.2, 1.0);
  target += -log(5) - 0.8 * log(abs(v_pow5));
  pow(v_pow1p5, 0.6666666666666666) ~ gamma(3, 2);
  target += -log(1.5) - 0.33333333333333337 * log(v_pow1p5);
  pow(v_powmhalf, -2) ~ gamma(3, 2);
  target += log(2) - 3 * log(v_powmhalf);
  z ~ normal(v_exp, 1);
}
