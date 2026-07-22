module {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.constant dense<3> : tensor<i32>
    %2 = stablehlo.constant dense<5> : tensor<i32>
    %3 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %9 = stablehlo.add %8, %7 : tensor<f32>
    %10 = chlo.lgamma %9 : tensor<f32> -> tensor<f32>
    %11 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %12 = stablehlo.convert %6 : (tensor<3xi32>) -> tensor<3xf32>
    %13 = stablehlo.add %12, %11 : tensor<3xf32>
    %14 = chlo.lgamma %13 : tensor<3xf32> -> tensor<3xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%14 init: %15) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.log %arg1 : tensor<3xf32>
    %19 = stablehlo.convert %6 : (tensor<3xi32>) -> tensor<3xf32>
    %20 = stablehlo.multiply %19, %18 : tensor<3xf32>
    %21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.add %10, %17 : tensor<f32>
    %24 = stablehlo.add %23, %22 : tensor<f32>
    return %24 : tensor<f32>
  }
}
