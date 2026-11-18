functions {
  real flatppl_code4(real lo, real hi, real alpha) {
    // pyhf code4 (§09 normsys_factor): 1 at alpha = 0, lo and hi at -1 and +1,
    // exponential tails, the C^2 polynomial inside.
    if (abs(alpha) >= 1) return exp(alpha > 0 ? alpha * log(hi) : -alpha * log(lo));
    real loglo = log(lo);
    real loghi = log(hi);
    real s0 = (hi + lo) / 2;
    real a0 = (hi - lo) / 2;
    real s1 = (hi * loghi - lo * loglo) / 2;
    real a1 = (hi * loghi + lo * loglo) / 2;
    real s2 = (hi * square(loghi) + lo * square(loglo)) / 2;
    real a2 = (hi * square(loghi) - lo * square(loglo)) / 2;
    real x = alpha;
    return 1 + x * ((15 * a0 - 7 * s1 + a2) / 8 + x * ((24 * (s0 - 1) - 9 * a1 + s2) / 8
           + x * ((-5 * a0 + 5 * s1 - a2) / 4 + x * ((-12 * (s0 - 1) + 7 * a1 - s2) / 4
           + x * ((3 * a0 - 3 * s1 + a2) / 8 + x * (8 * (s0 - 1) - 5 * a1 + s2) / 8)))));
  }
  vector flatppl_code4(vector lo, vector hi, real alpha) {
    // Elementwise in the anchors: the branch depends on alpha alone.
    int n = rows(lo);
    if (abs(alpha) >= 1) return exp(alpha > 0 ? alpha * log(hi) : -alpha * log(lo));
    vector[n] loglo = log(lo);
    vector[n] loghi = log(hi);
    vector[n] s0 = (hi + lo) / 2;
    vector[n] a0 = (hi - lo) / 2;
    vector[n] s1 = (hi .* loghi - lo .* loglo) / 2;
    vector[n] a1 = (hi .* loghi + lo .* loglo) / 2;
    vector[n] s2 = (hi .* square(loghi) + lo .* square(loglo)) / 2;
    vector[n] a2 = (hi .* square(loghi) - lo .* square(loglo)) / 2;
    real x = alpha;
    return 1 + x * ((15 * a0 - 7 * s1 + a2) / 8 + x * ((24 * (s0 - 1) - 9 * a1 + s2) / 8
           + x * ((-5 * a0 + 5 * s1 - a2) / 4 + x * ((-12 * (s0 - 1) + 7 * a1 - s2) / 4
           + x * ((3 * a0 - 3 * s1 + a2) / 8 + x * (8 * (s0 - 1) - 5 * a1 + s2) / 8)))));
  }
  vector flatppl_code4(vector lo, vector hi, vector alpha) {
    // One alpha per element. §09 evaluates the polynomial at the clamped alpha, so
    // on positive anchors both pieces are finite and a 0/1 mask selects per element.
    int n = rows(lo);
    vector[n] loglo = log(lo);
    vector[n] loghi = log(hi);
    vector[n] s0 = (hi + lo) / 2;
    vector[n] a0 = (hi - lo) / 2;
    vector[n] s1 = (hi .* loghi - lo .* loglo) / 2;
    vector[n] a1 = (hi .* loghi + lo .* loglo) / 2;
    vector[n] s2 = (hi .* square(loghi) + lo .* square(loglo)) / 2;
    vector[n] a2 = (hi .* square(loghi) - lo .* square(loglo)) / 2;
    vector[n] x = fmin(fmax(alpha, -1), 1);
    vector[n] poly = 1 + x .* ((15 * a0 - 7 * s1 + a2) / 8 + x .* ((24 * (s0 - 1) - 9 * a1 + s2) / 8
           + x .* ((-5 * a0 + 5 * s1 - a2) / 4 + x .* ((-12 * (s0 - 1) + 7 * a1 - s2) / 4
           + x .* ((3 * a0 - 3 * s1 + a2) / 8 + x .* (8 * (s0 - 1) - 5 * a1 + s2) / 8)))));
    vector[n] tail = exp(fmax(alpha, 0) .* loghi + fmax(-alpha, 0) .* loglo);
    vector[n] m = fmin(floor(abs(alpha)), 1);
    return m .* tail + (1 - m) .* poly;
  }
  real flatppl_interp_poly6_exp(real left, real center, real right, real alpha) {
    return center * flatppl_code4(left / center, right / center, alpha);
  }
  vector flatppl_interp_poly6_exp(vector left, vector center, vector right, real alpha) {
    return center .* flatppl_code4(left ./ center, right ./ center, alpha);
  }
  vector flatppl_interp_poly6_exp(vector left, vector center, vector right, vector alpha) {
    return center .* flatppl_code4(left ./ center, right ./ center, alpha);
  }
  real flatppl_code4p(real down, real up, real alpha) {
    // pyhf code4p (§09 histosys_shift): 0 at alpha = 0, -down and up at -1 and +1,
    // linear tails, the C^2 polynomial inside.
    if (abs(alpha) >= 1) return alpha * (alpha > 0 ? up : down);
    return alpha * ((up + down) / 2 + alpha * (up - down) * (15 - 10 * square(alpha) + 3 * alpha ^ 4) / 16);
  }
  vector flatppl_code4p(vector down, vector up, real alpha) {
    if (abs(alpha) >= 1) return alpha * (alpha > 0 ? up : down);
    return alpha * ((up + down) / 2 + alpha * (up - down) * (15 - 10 * square(alpha) + 3 * alpha ^ 4) / 16);
  }
  vector flatppl_code4p(vector down, vector up, vector alpha) {
    // One alpha per element. §09 evaluates the polynomial at the clamped alpha, so
    // both pieces are finite and a 0/1 mask selects per element.
    int n = rows(down);
    vector[n] x = fmin(fmax(alpha, -1), 1);
    vector[n] poly = x .* ((up + down) / 2 + x .* (up - down) .* (15 - 10 * square(x) + 3 * x .^ 4) / 16);
    vector[n] m = fmin(floor(abs(alpha)), 1);
    return m .* (fmax(alpha, 0) .* up + fmin(alpha, 0) .* down) + (1 - m) .* poly;
  }
  real flatppl_interp_poly6_lin(real left, real center, real right, real alpha) {
    return center + flatppl_code4p(center - left, right - center, alpha);
  }
  vector flatppl_interp_poly6_lin(vector left, vector center, vector right, real alpha) {
    return center + flatppl_code4p(center - left, right - center, alpha);
  }
  vector flatppl_interp_poly6_lin(vector left, vector center, vector right, vector alpha) {
    return center + flatppl_code4p(center - left, right - center, alpha);
  }
}
data {
  real y;
}
parameters {
  real alpha;
}
model {
  real scale = flatppl_interp_poly6_exp(0.95, 1.0, 1.05, alpha);
  real shift = flatppl_interp_poly6_lin(0.8, 1.0, 1.3, alpha);
  real expected = 20.0 * scale + shift;

  y ~ normal(expected, 1.0);
}
