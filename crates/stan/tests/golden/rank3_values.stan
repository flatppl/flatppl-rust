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
  vector g(array[] matrix t) {
    return t[1, 2, :]';
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_nom_1;
  int<lower=0> N_nom_2;
  int<lower=0> N_fa_1;
  int<lower=0> N_fa_2;
  int<lower=0> N_fa_3;
  int<lower=0> N_ob;
  vector[N] y;
  matrix[N_nom_1, N_nom_2] nom;
  array[N_fa_1] matrix[N_fa_2, N_fa_3] fa;
  vector[N_ob] ob;
}
transformed data {
  array[2, 1, 4] int fi = rep_array(1, 2, 1, 4);
  matrix[2, 4] y_2 = flatppl_sample_yields(nom, fa, flatppl_matrices(fi));
  matrix[2, 4] y4 = flatppl_sample_yields(nom, rep_array(rep_matrix(0.5, 3, 4), num_elements(ob)), flatppl_matrices(fi));
  matrix[N_fa_1, N_fa_3] sl = to_matrix(fa[:, 2, :]);
  vector[N_fa_3 + N_fa_2] v = append_row(fa[2, 3, :]', fa[1, :, 4]);
}
parameters {
  real mu;
  array[2] matrix[1, 4] th;
}
model {
  matrix[2, 4] y2 = flatppl_sample_yields(nom, fa, th);
  matrix[2, 4] y3 = flatppl_sample_yields(nom, fa, rep_array(rep_matrix(mu, 1, 4), 2));

  y ~ normal(append_row(append_row(append_row(append_row(append_row(append_row(y_2[1, :]', y2[2, :]'), y3[1, :]'), y4[2, :]'), sl[2, :]'), v), g(fa)), 1.0);
}
