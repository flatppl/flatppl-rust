module {
  func.func @logdensity(%arg0: tensor<2x3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
    %3 = stablehlo.constant dense<2.0> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %5 = stablehlo.divide %2, %4 : tensor<3xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
    %7 = stablehlo.subtract %0, %6 : tensor<3x2xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<3x2xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
    %11 = stablehlo.constant dense<1.0> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %13 = stablehlo.divide %10, %12 : tensor<3xf32>
    return %13 : tensor<3xf32>
  }
}
