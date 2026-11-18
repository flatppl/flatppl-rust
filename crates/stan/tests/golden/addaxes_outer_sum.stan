data {
  int<lower=0> N;
  int<lower=0> N_v;
  int<lower=0> T;
  int<lower=0> M;
  real z;
  vector[N] x;
  vector[N_v] v;
}
transformed data {
  matrix[N, N_v] d2 = (rep_matrix(x, N_v) - rep_matrix(v', N)) .^ 2;
}
parameters {
  vector<lower=0, upper=1>[T] mean_p;
  real<lower=0, upper=5> s;
  vector[M] e;
}
model {
  matrix[M, T] lp = rep_matrix(s * e, T) + rep_matrix(logit(mean_p)', M);
  matrix[M, T] q = e * mean_p';

  mean_p ~ uniform(0, 1);
  s ~ uniform(0, 5);
  e ~ std_normal();
  z ~ normal(sum(lp + q - d2), 1);
}
