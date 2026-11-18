data {
  int<lower=0> N;
  real z;
  int<lower=0> T;
  vector[N] x_obs;
}
parameters {
  real r;
}
model {
  real u = 1.0;
  real v = 0.0;
  for (t in 1:T) {
    real u_next = u + r * v + x_obs[t];
    real v_next = 0.5 * u - v;
    u = u_next;
    v = v_next;
  }

  r ~ std_normal();
  z ~ normal(u + v, 1);
}
