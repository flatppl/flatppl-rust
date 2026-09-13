module {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> tensor<f32> {
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.add %2, %1 : tensor<f32>
    %4 = chlo.lgamma %3 : tensor<f32> -> tensor<f32>
    %6 = stablehlo.constant dense<[2.0, 3.0, 5.0]> : tensor<3xf32>
    %8 = stablehlo.constant dense<[0.6931471824645996, 1.7917594909667969, 4.787491798400879]> : tensor<3xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.log %arg1 : tensor<3xf32>
    %13 = stablehlo.multiply %6, %12 : tensor<3xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%13 init: %14) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.add %4, %11 : tensor<f32>
    %17 = stablehlo.add %16, %15 : tensor<f32>
    return %17 : tensor<f32>
  }
}
