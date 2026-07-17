module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    return %0, %key : tensor<f32>, tensor<2xui64>
  }
}
