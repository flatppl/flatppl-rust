data {
  vector[3] y_obs;
}
parameters {
  vector[4] theta;
}
model {
  real mu = theta[1];
  real s = theta[2];
  vector[2] w = theta[{3, 4}];

  mu ~ normal(0, 2);
  s ~ std_normal() T[0.0, ];
  w ~ std_normal();
  y_obs ~ normal(mu + w[1] - w[2], s);
}
