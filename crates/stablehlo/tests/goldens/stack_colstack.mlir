module {
  func.func @logdensity() -> tensor<3x2xf32> {
    %2 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %3 : tensor<3x2xf32>
  }
}
