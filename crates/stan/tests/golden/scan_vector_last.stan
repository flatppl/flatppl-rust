data {
  int<lower=0> N;
  int<lower=0> K;
  real z;
  int<lower=0> T;
  vector[N] x_obs;
}
parameters {
  vector[K] w;
}
model {
  vector[K] last = zeros_vector(K);
  for (t in 1:T) {
    last = 0.5 * last + w * x_obs[t];
  }

  w ~ std_normal();
  z ~ normal(sum(last), 1);
}
