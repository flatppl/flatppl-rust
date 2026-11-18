data {
  real z;
}
parameters {
  real m;
  real<lower=0> sigma;
  real<lower=0> tau;
  real<lower=0.0> kappa;
}
model {
  m ~ normal(1, 2);
  sigma ~ student_t(3, 0, 3) T[0, ];
  tau ~ student_t(4, 1, 3) T[0, ];
  kappa ~ student_t(5, m, 2) T[0.0, ];
  z ~ normal(m, sigma + tau + kappa);
}
