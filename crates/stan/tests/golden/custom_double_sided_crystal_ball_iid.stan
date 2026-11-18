functions {
  real flatppl_double_sided_crystal_ball_lpdf(real x, real m0, real sigmaL, real sigmaR,
                                                real alphaL, real alphaR, real nL, real nR) {
    real aL = abs(alphaL);
    real aR = abs(alphaR);
    real log_left = log(sigmaL) + log_sum_exp(log(nL) - log(aL) - log(nL - 1) - 0.5 * square(aL),
                                              0.5 * log(pi() / 2) + log(erf(aL / sqrt2())));
    real log_right = log(sigmaR) + log_sum_exp(log(nR) - log(aR) - log(nR - 1) - 0.5 * square(aR),
                                               0.5 * log(pi() / 2) + log(erf(aR / sqrt2())));
    real log_norm = log_sum_exp(log_left, log_right);
    if (x <= m0) {
      real t = (x - m0) / sigmaL;
      if (t < -aL) return nL * log(nL / aL) - 0.5 * square(aL) - nL * log(nL / aL - aL - t) - log_norm;
      return -0.5 * square(t) - log_norm;
    }
    real t = (x - m0) / sigmaR;
    if (t > aR) return nR * log(nR / aR) - 0.5 * square(aR) - nR * log(nR / aR - aR + t) - log_norm;
    return -0.5 * square(t) - log_norm;
  }
  real flatppl_double_sided_crystal_ball_lpdf(vector x, real m0, real sigmaL, real sigmaR,
                                              real alphaL, real alphaR, real nL, real nR) {
    // Piecewise in x: the normalizer once, the branch per element.
    real aL = abs(alphaL);
    real aR = abs(alphaR);
    real log_left = log(sigmaL) + log_sum_exp(log(nL) - log(aL) - log(nL - 1) - 0.5 * square(aL),
                                              0.5 * log(pi() / 2) + log(erf(aL / sqrt2())));
    real log_right = log(sigmaR) + log_sum_exp(log(nR) - log(aR) - log(nR - 1) - 0.5 * square(aR),
                                               0.5 * log(pi() / 2) + log(erf(aR / sqrt2())));
    real lp = -num_elements(x) * log_sum_exp(log_left, log_right);
    for (i in 1:num_elements(x)) {
      if (x[i] <= m0) {
        real t = (x[i] - m0) / sigmaL;
        lp += t < -aL ? nL * log(nL / aL) - 0.5 * square(aL) - nL * log(nL / aL - aL - t) : -0.5 * square(t);
      } else {
        real t = (x[i] - m0) / sigmaR;
        lp += t > aR ? nR * log(nR / aR) - 0.5 * square(aR) - nR * log(nR / aR - aR + t) : -0.5 * square(t);
      }
    }
    return lp;
  }
}
data {
  vector[6] y_obs;
}
parameters {
  real m;
}
model {
  m ~ std_normal();
  y_obs ~ flatppl_double_sided_crystal_ball(m, 1.0, 0.7, 1.2, 1.8, 2.5, 4.0);
}
