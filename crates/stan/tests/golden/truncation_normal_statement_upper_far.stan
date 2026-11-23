functions {
  real flatppl_lphi(real z) {
    // log Phi(z) from erfc for z >= -37: Stan's normal_lcdf derivative is an approximation.
    if (z > 0) {
      return log1m(0.5 * erfc(z / sqrt2()));
    }
    if (z >= -37) {
      return log(erfc(-z / sqrt2())) - log2();
    }
    return std_normal_lcdf(z);
  }
}
data {
  real y;
}
parameters {
  real mu;
}
model {
  if (y > -10.0) {
    target += negative_infinity();
  } else {
    y ~ normal(mu, 1.0);
    target += -flatppl_lphi(-10 - mu);
  }
}
