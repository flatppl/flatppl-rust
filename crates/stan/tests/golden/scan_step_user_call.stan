functions {
  real scale(real p_2) {
    return p_2;
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_u_data;
  vector[N] y;
  vector[N_u_data] u_data;
}
parameters {
  real omega;
}
model {
  vector[N_u_data] states_m;
  {
    real m = 0;
    real p = 1;
    for (t in 1:N_u_data) {
      real u = u_data[t];
      real g = (omega / scale(p));
      m = m + g * (u - m);
      p = p + 1.0;
      states_m[t] = m;
    }
  }

  omega ~ std_normal();
  y ~ normal(states_m, 1.0);
}
