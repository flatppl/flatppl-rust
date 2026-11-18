data {
  vector[3] y_obs;
}
parameters {
  vector<lower=0>[3] s;
}
model {
  s ~ normal(0, 2) T[0, ];
  y_obs ~ normal(0.0, s);
}
