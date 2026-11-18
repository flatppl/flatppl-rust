functions {
  real flatppl_interp_poly6_exp(real left, real center, real right, real alpha) {
    // §09: center times pyhf code4 with anchors left / center and right / center.
    real lo = left / center;
    real hi = right / center;
    if (abs(alpha) >= 1) return center * exp(alpha > 0 ? alpha * log(hi) : -alpha * log(lo));
    real loglo = log(lo);
    real loghi = log(hi);
    real s0 = (hi + lo) / 2;
    real a0 = (hi - lo) / 2;
    real s1 = (hi * loghi - lo * loglo) / 2;
    real a1 = (hi * loghi + lo * loglo) / 2;
    real s2 = (hi * square(loghi) + lo * square(loglo)) / 2;
    real a2 = (hi * square(loghi) - lo * square(loglo)) / 2;
    real c1 = (15 * a0 - 7 * s1 + a2) / 8;
    real c2 = (24 * (s0 - 1) - 9 * a1 + s2) / 8;
    real c3 = (-5 * a0 + 5 * s1 - a2) / 4;
    real c4 = (-12 * (s0 - 1) + 7 * a1 - s2) / 4;
    real c5 = (3 * a0 - 3 * s1 + a2) / 8;
    real c6 = (8 * (s0 - 1) - 5 * a1 + s2) / 8;
    real x = alpha;
    return center * (1 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6))))));
  }
  vector flatppl_interp_poly6_exp(vector left, vector center, vector right, real alpha) {
    // Elementwise in the anchors: the branch depends on alpha alone.
    int n = rows(left);
    vector[n] lo = left ./ center;
    vector[n] hi = right ./ center;
    if (abs(alpha) >= 1) return center .* exp(alpha > 0 ? alpha * log(hi) : -alpha * log(lo));
    vector[n] loglo = log(lo);
    vector[n] loghi = log(hi);
    vector[n] s0 = (hi + lo) / 2;
    vector[n] a0 = (hi - lo) / 2;
    vector[n] s1 = (hi .* loghi - lo .* loglo) / 2;
    vector[n] a1 = (hi .* loghi + lo .* loglo) / 2;
    vector[n] s2 = (hi .* square(loghi) + lo .* square(loglo)) / 2;
    vector[n] a2 = (hi .* square(loghi) - lo .* square(loglo)) / 2;
    vector[n] c1 = (15 * a0 - 7 * s1 + a2) / 8;
    vector[n] c2 = (24 * (s0 - 1) - 9 * a1 + s2) / 8;
    vector[n] c3 = (-5 * a0 + 5 * s1 - a2) / 4;
    vector[n] c4 = (-12 * (s0 - 1) + 7 * a1 - s2) / 4;
    vector[n] c5 = (3 * a0 - 3 * s1 + a2) / 8;
    vector[n] c6 = (8 * (s0 - 1) - 5 * a1 + s2) / 8;
    real x = alpha;
    return center .* (1 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6))))));
  }
  real flatppl_interp_poly6_lin(real left, real center, real right, real alpha) {
    // §09: center plus pyhf code4p with down = center - left and up = right - center.
    real down = center - left;
    real up = right - center;
    if (abs(alpha) >= 1) return center + alpha * (alpha > 0 ? up : down);
    return center + alpha * ((up + down) / 2
                             + alpha * (up - down) * (15 - 10 * square(alpha) + 3 * alpha ^ 4) / 16);
  }
  vector flatppl_interp_poly6_lin(vector left, vector center, vector right, real alpha) {
    // Elementwise in the anchors: the branch depends on alpha alone.
    vector[rows(left)] down = center - left;
    vector[rows(left)] up = right - center;
    if (abs(alpha) >= 1) return center + alpha * (alpha > 0 ? up : down);
    return center + alpha * ((up + down) / 2
                             + alpha * (up - down) * (15 - 10 * square(alpha) + 3 * alpha ^ 4) / 16);
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
