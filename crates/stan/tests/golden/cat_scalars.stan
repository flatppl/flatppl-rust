data {
  vector[3] ys;
}
transformed data {
  array[3] int ks = {1, 2, 3};
}
parameters {
  real mu;
}
model {
  vector[3] v = [0.5, mu, 2.5]';

  mu ~ std_normal();
  ys ~ normal(v .* to_vector(ks), 1);
}
