module {
  func.func @sample(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<2> : tensor<1xi64>
    %4 = stablehlo.rng %1, %2, %3, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<2xf32>
    %5 = stablehlo.dot_general %0, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<2xf32>
    return %6 : tensor<2xf32>
  }
}
