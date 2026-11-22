data {
  real m_theta;
}
parameters {
  real theta;
}
model {
  theta ~ normal(m_theta, 1.0);
}
