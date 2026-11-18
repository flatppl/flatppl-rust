data {
  int<lower=0> N;
  vector[N] y_obs;
  vector[N] x;
  vector[N] se;
}
transformed data {
  for (i in 1:N) {
    if (is_inf(se[i]) || is_nan(se[i])) {
      reject("se[", i, "] is not finite");
    }
  }
}
parameters {
  real mu;
  real<lower=0> sigma;
  real<lower=0> nu;
}
model {
  vector[N] s = sqrt(se .* se + sigma * sigma);

  mu ~ std_normal();
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);
  y_obs ~ student_t(nu, mu + x, s);
}
