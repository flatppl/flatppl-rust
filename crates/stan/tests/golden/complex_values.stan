functions {
  complex wave(real x, real y_2) {
    return to_complex(x, y_2);
  }
}
data {
  real y;
}
parameters {
  real a;
  real b;
}
model {
  complex z = wave(a, b);
  complex_vector[2] v = [z, 2 + 1i]';
  complex_row_vector[2] row = v';
  complex_row_vector[2] conjugated = conj(v');
  complex_matrix[2, 2] A = append_row(v', v');
  array[2] complex_matrix[2, 2] blocks = {A, A};
  real location = get_real((to_matrix(row) * v)[1]) + 2 * get_real((to_matrix(conjugated) * v)[1]) + get_imag(blocks[2, 1, 2]) + get_imag(conj(z)) + norm(z);

  a ~ std_normal();
  b ~ std_normal();
  y ~ normal(location, 1);
}
