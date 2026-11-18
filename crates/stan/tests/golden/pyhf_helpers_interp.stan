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
  real flatppl_histosys_shift(real lo, real nominal, real hi, real alpha) {
    return flatppl_code4p(nominal - lo, hi - nominal, alpha);
  }
  vector flatppl_histosys_shift(vector lo, vector nominal, vector hi, real alpha) {
    return flatppl_code4p(nominal - lo, hi - nominal, alpha);
  }
  vector flatppl_histosys_shift(vector lo, vector nominal, vector hi, vector alpha) {
    return flatppl_code4p(nominal - lo, hi - nominal, alpha);
  }
  matrix flatppl_histosys_shift(matrix lo, matrix nominal, matrix hi, vector alpha) {
    // Row m takes alpha[m]: the per-element form over the column-major entries.
    int r = rows(lo);
    int c = cols(lo);
    return to_matrix(flatppl_code4p(to_vector(nominal - lo), to_vector(hi - nominal),
                                    to_vector(rep_matrix(alpha, c))), r, c);
  }
}
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
}
model {
  real nf = flatppl_code4(0.9, 1.15, alpha);
  real hs = flatppl_histosys_shift(46.0, 50.0, 56.0, alpha);
  vector[2] nfv = flatppl_code4([0.9, 0.8]', [1.15, 1.3]', [alpha, beta]');
  matrix[2, 2] hsm = flatppl_histosys_shift(append_row([46.0, 90.0], [5.0, 4.0]), append_row([50.0, 100.0], [4.0, 4.5]), append_row([56.0, 120.0], [2.5, 6.0]), [alpha, beta]');
  vector[2] hsv = flatppl_histosys_shift(rep_vector(46.0, 2), [50.0, 52.0]', rep_vector(56.0, 2), [alpha, beta]');

  y ~ normal(append_row(append_row(append_row(append_row([nf, hs]', nfv), hsm[1, :]'), hsm[2, :]'), hsv), 1.0);
}
