data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
  vector[N] se;
}
transformed data {
  for (i in 1:N) {
    if (!(se[i] > 0) || is_inf(se[i])) {
      reject("normalize(weighted(c, M)) has weight 0 at se[", i, "]");
    }
  }
}
parameters {
  real mu;
  real<lower=0> nu;
}
model {
  mu ~ std_normal();
  nu ~ gamma(2, 0.1);
  y_obs ~ student_t(nu, mu + x, se);
}
