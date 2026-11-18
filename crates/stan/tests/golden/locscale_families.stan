data {
  real z;
}
parameters {
  real m;
  real<lower=0> s;
  real a;
  real b;
  real c;
  real d;
}
model {
  m ~ std_normal();
  s ~ lognormal(0, 1);
  a ~ normal(m, 2);
  b ~ cauchy(1 + s * 0.5, s * 2);
  c ~ logistic(m, s);
  d ~ double_exponential(1, 1);
  z ~ normal(a + b + c + d, 1);
}
