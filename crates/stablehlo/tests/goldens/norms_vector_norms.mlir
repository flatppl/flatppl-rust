module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.abs %arg0 : tensor<4xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %3 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%3 init: %4) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %6 = stablehlo.sqrt %5 : tensor<f32>
    %7 = stablehlo.abs %arg0 : tensor<4xf32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%7 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %11 = stablehlo.divide %arg0, %10 : tensor<4xf32>
    %12 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    %13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.sqrt %14 : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.divide %arg0, %16 : tensor<4xf32>
    %18 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %19 = stablehlo.reduce(%arg0 init: %18) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %21 = stablehlo.subtract %arg0, %20 : tensor<4xf32>
    %22 = stablehlo.exponential %21 : tensor<4xf32>
    %23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%22 init: %23) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %26 = stablehlo.divide %22, %25 : tensor<4xf32>
    %27 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %28 = stablehlo.reduce(%arg0 init: %27) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %30 = stablehlo.subtract %arg0, %29 : tensor<4xf32>
    %31 = stablehlo.exponential %30 : tensor<4xf32>
    %32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.reduce(%31 init: %32) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %34 = stablehlo.log %33 : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %36 = stablehlo.subtract %30, %35 : tensor<4xf32>
    return %2, %6, %11, %17, %26, %36 : tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
