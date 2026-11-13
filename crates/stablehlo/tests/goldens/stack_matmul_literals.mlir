module {
  func.func @logdensity() -> tensor<2x2xi32> {
    %2 = stablehlo.constant dense<[[1, 3, 5], [9, 5, 1]]> : tensor<2x3xi32>
    %3 = stablehlo.reshape %2 : (tensor<2x3xi32>) -> tensor<2x3xi32>
    %7 = stablehlo.constant dense<[[1, 0], [0, 1], [1, 1]]> : tensor<3x2xi32>
    %8 = stablehlo.reshape %7 : (tensor<3x2xi32>) -> tensor<3x2xi32>
    %9 = stablehlo.dot_general %3, %8, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<2x3xi32>, tensor<3x2xi32>) -> tensor<2x2xi32>
    return %9 : tensor<2x2xi32>
  }
}
