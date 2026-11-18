data {
  real z;
}
parameters {
  real mu;
}
model {
  vector[6] v = append_row(append_row([0.5, mu]', [1.0]'), [mu, 2.0, -1.0]');
  array[4] vector[2] R = append_array(append_array({[1.0, mu]'}, {[0.5, 0.5]', [2.0, -1.0]'}), {[mu, mu]'});
  array[2] vector[2] S2 = append_array({[mu, 1.0]'}, {[2.0, mu]'});

  mu ~ std_normal();
  z ~ normal(sum(v) + R[4][1] + R[2][2] + S2[2][1], 1);
}
