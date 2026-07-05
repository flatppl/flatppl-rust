module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<5.0> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.add %arg0, %7 : tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %11 = stablehlo.add %6, %10 : tensor<3xf32>
    %12 = chlo.lgamma %11 : tensor<3xf32> -> tensor<3xf32>
    %13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.log %arg1 : tensor<3xf32>
    %17 = stablehlo.multiply %6, %16 : tensor<3xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.reduce(%17 init: %18) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.add %9, %15 : tensor<f32>
    %21 = stablehlo.add %20, %19 : tensor<f32>
    return %21 : tensor<f32>
  }
}
