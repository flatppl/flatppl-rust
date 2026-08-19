module {
  func.func @logdensity(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<3xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<2x3xf32>) -> tensor<2x2x3xf32>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<2x3xf32>) -> tensor<2x2x3xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<3xf32>) -> tensor<2x2x3xf32>
    %4 = stablehlo.subtract %0, %2 : tensor<2x2x3xf32>
    %5 = stablehlo.constant dense<2> : tensor<i32>
    %6 = stablehlo.convert %5 : (tensor<i32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2x2x3xf32>
    %8 = stablehlo.power %4, %7 : tensor<2x2x3xf32>
    %9 = stablehlo.multiply %8, %3 : tensor<2x2x3xf32>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%9 init: %10) applies stablehlo.add across dimensions = [2] : (tensor<2x2x3xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %11 : tensor<2x2xf32>
  }
}
