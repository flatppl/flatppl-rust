module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.abs %arg0 : tensor<4xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %3 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%3 init: %4) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %6 = stablehlo.sqrt %5 : tensor<f32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%0 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %10 = stablehlo.divide %arg0, %9 : tensor<4xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%3 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %13 = stablehlo.sqrt %12 : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %15 = stablehlo.divide %arg0, %14 : tensor<4xf32>
    %16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.reduce(%arg0 init: %16) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %19 = stablehlo.subtract %arg0, %18 : tensor<4xf32>
    %20 = stablehlo.exponential %19 : tensor<4xf32>
    %21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %24 = stablehlo.divide %20, %23 : tensor<4xf32>
    %25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %26 = stablehlo.reduce(%arg0 init: %25) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %28 = stablehlo.subtract %arg0, %27 : tensor<4xf32>
    %29 = stablehlo.exponential %28 : tensor<4xf32>
    %30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.reduce(%29 init: %30) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.log %31 : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %34 = stablehlo.subtract %28, %33 : tensor<4xf32>
    return %2, %6, %10, %15, %24, %34 : tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
