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
    %8 = stablehlo.divide %7, %4 : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %10 = stablehlo.subtract %arg0, %9 : tensor<4xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<4xf32>
    %12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%11 init: %12) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %14 = stablehlo.constant dense<3.0> : tensor<f32>
    %15 = stablehlo.divide %13, %14 : tensor<f32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.reduce(%arg0 init: %16) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.divide %17, %4 : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %20 = stablehlo.subtract %arg0, %19 : tensor<4xf32>
    %21 = stablehlo.multiply %20, %20 : tensor<4xf32>
    %22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = stablehlo.reduce(%21 init: %22) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.divide %23, %14 : tensor<f32>
    %25 = stablehlo.sqrt %24 : tensor<f32>
    return %1, %5, %15, %25 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
