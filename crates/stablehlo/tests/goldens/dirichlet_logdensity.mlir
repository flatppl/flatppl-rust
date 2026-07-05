module {
  func.func @logdensity(%arg0: tensor<3xf32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg0 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = chlo.lgamma %arg0 : tensor<3xf32> -> tensor<3xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %15 = stablehlo.subtract %arg0, %14 : tensor<3xf32>
    %16 = stablehlo.log %6 : tensor<3xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<3xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.reduce(%17 init: %18) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.add %9, %13 : tensor<f32>
    %21 = stablehlo.add %20, %19 : tensor<f32>
    return %21 : tensor<f32>
  }
}
