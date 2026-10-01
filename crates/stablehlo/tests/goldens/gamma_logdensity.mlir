module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.log %arg1 : tensor<f32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<f32>
    %6 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.subtract %arg0, %3 : tensor<f32>
    %9 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.multiply %arg1, %0 : tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %5, %7 : tensor<f32>
    %14 = stablehlo.add %13, %10 : tensor<f32>
    %15 = stablehlo.add %14, %12 : tensor<f32>
    return %15 : tensor<f32>
  }
}
