data {
  real y;
}
model {
  y ~ std_normal() T[3.9, 10.0];
}
