module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.log %arg2 : tensor<f32>
    %2 = stablehlo.constant dense<2.0> : tensor<f32>
    %3 = stablehlo.multiply %2, %arg1 : tensor<f32>
    %4 = stablehlo.log %3 : tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.divide %6, %arg2 : tensor<f32>
    %8 = chlo.lgamma %7 : tensor<f32> -> tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = stablehlo.subtract %0, %arg0 : tensor<f32>
    %11 = stablehlo.abs %10 : tensor<f32>
    %12 = stablehlo.divide %11, %arg1 : tensor<f32>
    %13 = stablehlo.power %12, %arg2 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.add %1, %5 : tensor<f32>
    %16 = stablehlo.add %15, %9 : tensor<f32>
    %17 = stablehlo.add %16, %14 : tensor<f32>
    return %17 : tensor<f32>
  }
}
