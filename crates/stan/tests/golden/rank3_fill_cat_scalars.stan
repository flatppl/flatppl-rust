functions {
  array[] matrix flatppl_matrices(array[,,] int x) {
    // §09 promotes integer entries to real: Stan converts one array[,] at a time.
    array[size(x)] matrix[size(x[1]), size(x[1, 1])] out;
    for (s in 1:size(x)) {
      out[s] = to_matrix(x[s]);
    }
    return out;
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
}
data {
  vector[13] y;
  real c;
}
transformed data {
  array[1, 1, 2] int z = rep_array((-1), 1, 1, 2);
  array[1] matrix[1, 2] f = rep_array(rep_matrix(c, 1, 2), 1);
  matrix[1, 2] y_2 = flatppl_sample_yields(to_matrix([10.0, 20.0]), flatppl_matrices(z), f);
}
parameters {
  real a;
  real mu;
}
model {
  vector[4] u = [a, mu, 1.0, 2.0]';
  vector[4] w = [3.0, 4.0, a, mu]';
  vector[3] s = [a, mu, 1.5]';

  y ~ normal(append_row(append_row(append_row(y_2[1, :]', u), w), s), 1.0);
}
