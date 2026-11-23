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
  real v;
}
parameters {
  real a;
}
model {
  a ~ std_normal();
  target += (v < 0.0 ? negative_infinity() : normal_lpdf(v | a, 1) - flatppl_lphi(a));
}
