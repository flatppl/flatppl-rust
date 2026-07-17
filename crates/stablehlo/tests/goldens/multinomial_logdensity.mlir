module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.constant dense<3> : tensor<i32>
    %2 = stablehlo.constant dense<5> : tensor<i32>
    %3 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.add %arg0, %7 : tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %11 = stablehlo.convert %6 : (tensor<3xi32>) -> tensor<3xf32>
    %12 = stablehlo.add %11, %10 : tensor<3xf32>
    %13 = chlo.lgamma %12 : tensor<3xf32> -> tensor<3xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%13 init: %14) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.log %arg1 : tensor<3xf32>
    %18 = stablehlo.convert %6 : (tensor<3xi32>) -> tensor<3xf32>
    %19 = stablehlo.multiply %18, %17 : tensor<3xf32>
    %20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%19 init: %20) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.add %9, %16 : tensor<f32>
    %23 = stablehlo.add %22, %21 : tensor<f32>
    return %23 : tensor<f32>
  }
}
