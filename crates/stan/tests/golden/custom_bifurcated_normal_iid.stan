functions {
  real flatppl_bifurcated_normal_lpdf(real x, real mu, real sigmaL, real sigmaR) {
    real s = x < mu ? sigmaL : sigmaR;
    return 0.5 * log(2 / pi()) - log(sigmaL + sigmaR) - 0.5 * square((x - mu) / s);
  }
  real flatppl_bifurcated_normal_lpdf(vector x, real mu, real sigmaL, real sigmaR) {
    // Piecewise in x: the width per element.
    real lp = num_elements(x) * (0.5 * log(2 / pi()) - log(sigmaL + sigmaR));
    for (i in 1:num_elements(x)) {
      real s = x[i] < mu ? sigmaL : sigmaR;
      lp -= 0.5 * square((x[i] - mu) / s);
    }
    return lp;
  }
}
data {
  vector[5] y_obs;
}
parameters {
  real mu;
}
model {
  mu ~ std_normal();
  y_obs ~ flatppl_bifurcated_normal(mu, 0.8, 1.6);
}
