module {
  func.func @logdensity() -> tensor<3x2xf32> {
    %3 = stablehlo.constant dense<[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]> : tensor<3x2xf32>
    %4 = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %5 = stablehlo.reshape %4 : (tensor<3xf32>) -> tensor<3x1xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %7 = stablehlo.multiply %3, %6 : tensor<3x2xf32>
    return %7 : tensor<3x2xf32>
  }
}
