data {
  vector[3] y_obs;
}
transformed data {
  real mu = 0.5;
  real s = 1.2;
  vector[2] w = [0.1, -0.2]';
}
model {
  mu ~ normal(0, 2);
  s ~ std_normal() T[0.0, ];
  w ~ std_normal();
  y_obs ~ normal(mu + w[1] - w[2], s);
}
