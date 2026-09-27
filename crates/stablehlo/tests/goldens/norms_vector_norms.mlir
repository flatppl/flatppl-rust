module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.abs %arg0 : tensor<4xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %3 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    %4 = stablehlo.reduce(%3 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.sqrt %4 : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.divide %arg0, %6 : tensor<4xf32>
    %8 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %9 = stablehlo.divide %arg0, %8 : tensor<4xf32>
    %10 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.reduce(%arg0 init: %10) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.subtract %arg0, %12 : tensor<4xf32>
    %14 = stablehlo.exponential %13 : tensor<4xf32>
    %15 = stablehlo.reduce(%14 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.divide %14, %16 : tensor<4xf32>
    %18 = stablehlo.log %15 : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %20 = stablehlo.subtract %13, %19 : tensor<4xf32>
    return %2, %5, %7, %9, %17, %20 : tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
