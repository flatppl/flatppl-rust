module {
  func.func @logdensity() -> tensor<2x2xi32> {
    %2 = stablehlo.constant dense<[[1, 3, 5], [9, 5, 1]]> : tensor<2x3xi32>
    %6 = stablehlo.constant dense<[[1, 0], [0, 1], [1, 1]]> : tensor<3x2xi32>
    %7 = stablehlo.dot_general %2, %6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xi32>, tensor<3x2xi32>) -> tensor<2x2xi32>
    return %7 : tensor<2x2xi32>
  }
}
