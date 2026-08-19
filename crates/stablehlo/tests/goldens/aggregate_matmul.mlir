module {
  func.func @logdensity(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<2x3xf32>) -> tensor<2x2x3xf32>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<2x3xf32>) -> tensor<2x2x3xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<2x2x3xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%3 init: %4) applies stablehlo.add across dimensions = [2] : (tensor<2x2x3xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %5 : tensor<2x2xf32>
  }
}
