functions {
  real flatppl_continued_poisson_lpdf(real x, real rate) {
    if (x < 0) return negative_infinity();
    return lmultiply(x, rate) - rate - lgamma(x + 1);
  }
  real flatppl_continued_poisson_lpdf(vector x, vector rate) {
    if (num_elements(x) > 0 && min(x) < 0) return negative_infinity();
    return sum(lmultiply(x, rate) - rate - lgamma(x + 1));
  }
  real flatppl_continued_poisson_lpdf(vector x, real rate) {
    if (num_elements(x) > 0 && min(x) < 0) return negative_infinity();
    return sum(lmultiply(x, rate)) - num_elements(x) * rate - sum(lgamma(x + 1));
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
  matrix flatppl_sample_yields(matrix nominal, array[] matrix shifts, array[] matrix factors) {
    // §09: (nominal + shifts summed over a) .* factors multiplied over m, per sample.
    // The sample index s reaches each matrix of an array: Stan maps no operation
    // over an array of matrices. The factor index m is a product over rows: Stan has
    // no column-wise product, and exp(sum(log(...))) fails at the zero factors §09 allows.
    matrix[rows(nominal), cols(nominal)] out;
    for (s in 1:rows(nominal)) {
      row_vector[cols(nominal)] y = nominal[s] + rep_row_vector(1, rows(shifts[s])) * shifts[s];
      for (m in 1:rows(factors[s])) {
        y = y .* factors[s][m];
      }
      out[s] = y;
    }
    return out;
  }
  vector flatppl_expected_counts(matrix samples) {
    // §09: the sum over the sample rows, as one row-vector product.
    return (rep_row_vector(1, rows(samples)) * samples)';
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_c_a_nominal;
  int<lower=0> N_c_b_nominal;
  vector[N] c_observed;
  vector[N_c_a_nominal] c_a_nominal;
  vector[N_c_b_nominal] c_b_nominal;
  real y;
}
transformed data {
  array[1] matrix[1, 2] c_factors_2 = rep_array(rep_matrix(1.0, 1, 2), 1);
}
parameters {
  real mu;
  real t;
}
model {
  matrix[2, 2] c_histosys_shifts = flatppl_histosys_shift(append_row([18.0, 9.0], [45.0, 55.0]), append_row(c_a_nominal', c_b_nominal'), append_row([22.0, 11.0], [56.0, 64.0]), [t, t]');
  matrix[1, 2] c_yields_site = c_histosys_shifts[{1}, :];
  vector[2] c_yields_agg = (rep_row_vector(1, rows(c_yields_site)) * c_yields_site)';
  array[1] matrix[1, 2] c_factors = {to_matrix(rep_vector(mu, 2), 1, 2, 0)};
  matrix[1, 2] c_yields = flatppl_sample_yields(to_matrix(c_a_nominal'), {to_matrix(c_yields_agg, 1, 2, 0)}, c_factors);
  matrix[1, 2] c_yields_2_site = c_histosys_shifts[{2}, :];
  vector[2] c_yields_2_agg = (rep_row_vector(1, rows(c_yields_2_site)) * c_yields_2_site)';
  matrix[1, 2] c_yields_2 = flatppl_sample_yields(to_matrix(c_b_nominal'), {to_matrix(c_yields_2_agg, 1, 2, 0)}, c_factors_2);
  vector[2] c_expected = flatppl_expected_counts(append_row(c_yields[1, :], c_yields_2[1, :]));

  target += flatppl_continued_poisson_lpdf(c_observed | c_expected);
  y ~ normal(t, 1.0);
}
