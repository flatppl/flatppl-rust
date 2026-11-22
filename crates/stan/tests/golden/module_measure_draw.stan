functions {
  real f_a(real par, int c_scaling) {
    return c_scaling * par;
  }
  real f_b(real arg1, real arg2) {
    return abs(arg1) * arg2;
  }
}
data {
  vector[10] obs;
  int<lower=0> c_scaling;
}
parameters {
  real theta1;
  real<lower=0> theta2;
}
model {
  real a = f_a(theta2, c_scaling);
  real b = f_b(theta1, theta2);

  theta1 ~ std_normal();
  theta2 ~ exponential(1);
  obs ~ normal(a, b);
}
