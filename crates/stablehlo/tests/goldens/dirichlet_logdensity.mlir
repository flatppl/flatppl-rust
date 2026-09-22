module {
  func.func @logdensity(%arg0: tensor<3xf32>) -> tensor<f32> {
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg0 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %3 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %4 = chlo.lgamma %arg0 : tensor<3xf32> -> tensor<3xf32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%4 init: %5) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<3xf32>
    %10 = stablehlo.constant dense<[-1.6094379425048828, -1.2039728164672852, -0.6931471824645996]> : tensor<3xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<3xf32>
    %12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%11 init: %12) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %14 = stablehlo.add %3, %7 : tensor<f32>
    %15 = stablehlo.add %14, %13 : tensor<f32>
    return %15 : tensor<f32>
  }
}
