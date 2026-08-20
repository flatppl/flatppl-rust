module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.multiply across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%arg0 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %4 = stablehlo.constant dense<4.0> : tensor<f32>
    %5 = stablehlo.divide %3, %4 : tensor<f32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%arg0 init: %6) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<4.0> : tensor<f32>
    %9 = stablehlo.divide %7, %8 : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %11 = stablehlo.subtract %arg0, %10 : tensor<4xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<4xf32>
    %13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.constant dense<3.0> : tensor<f32>
    %16 = stablehlo.divide %14, %15 : tensor<f32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%arg0 init: %17) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %19 = stablehlo.constant dense<4.0> : tensor<f32>
    %20 = stablehlo.divide %18, %19 : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %22 = stablehlo.subtract %arg0, %21 : tensor<4xf32>
    %23 = stablehlo.multiply %22, %22 : tensor<4xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.reduce(%23 init: %24) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %26 = stablehlo.constant dense<3.0> : tensor<f32>
    %27 = stablehlo.divide %25, %26 : tensor<f32>
    %28 = stablehlo.sqrt %27 : tensor<f32>
    return %1, %5, %16, %28 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
