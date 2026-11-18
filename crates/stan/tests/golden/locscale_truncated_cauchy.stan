data {
  real z;
}
parameters {
  real temp_Intercept;
  real<lower=0.0> sigma;
}
model {
  temp_Intercept ~ normal(1, 3);
  sigma ~ cauchy(temp_Intercept, 1.7) T[0.0, ];
  z ~ normal(temp_Intercept, sigma);
}
