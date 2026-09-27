module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.multiply across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%arg0 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %4 = stablehlo.constant dense<4.0> : tensor<f32>
    %5 = stablehlo.divide %3, %4 : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<4xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<4xf32>
    %9 = stablehlo.reduce(%8 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.constant dense<3.0> : tensor<f32>
    %11 = stablehlo.divide %9, %10 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    return %1, %5, %11, %12 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
