data {
  int<lower=0> N;
  array[N] int n;
}
parameters {
  vector[3] g;
}
model {
  g ~ std_normal();
  if (sum(n) != 10) {
    target += negative_infinity();
  } else {
    n ~ multinomial_logit(g);
  }
}
