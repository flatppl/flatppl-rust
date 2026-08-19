module {
  func.func @logdensity(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>, tensor<f32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %1 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %3 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %4 = stablehlo.select %2, %arg0, %3 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%4 init: %5) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %10 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %11 = stablehlo.compare EQ, %9, %10 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %12 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %13 = stablehlo.select %11, %arg0, %12 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%13 init: %14) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %16 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %17 = stablehlo.dot_general %arg1, %16, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
    %18 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
    %19 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<3x3xf32>
    %21 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %22 = stablehlo.dot_general %arg0, %21, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %23 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %24 = stablehlo.dot_general %23, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    return %8, %15, %17, %20, %22, %24 : tensor<f32>, tensor<3xf32>, tensor<f32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>
  }
}
