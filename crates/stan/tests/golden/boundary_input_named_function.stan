functions {
  real f(real a) {
    return 2.0 * a + 1.0;
  }
  real g(real z) {
    return f(z) - 1.0;
  }
}
parameters {
  real x;
}
model {
  x ~ normal(g(0.5), 1.0);
}
