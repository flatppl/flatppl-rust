data {
  int<lower=1> n;
}
transformed data {
  if (3 != n) {
    reject("the size of c_obs is ", 3, ", but its law gives ", n);
  }
}
parameters {
  real<lower=0> eta;
  matrix[3, 3] c_obs;
}
model {
  c_obs ~ lkj_corr(eta);
}
