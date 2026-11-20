parameters {
  real mu;
  real y;
}
model {
  target += log_sum_exp([log(0.5) + (y < -pi() || y > pi() ? negative_infinity() : von_mises_lpdf(y | mu, 2.0)), log(0.5) + normal_lpdf(y | mu, 1.0)]') - log(0.5 + 0.5);
}
