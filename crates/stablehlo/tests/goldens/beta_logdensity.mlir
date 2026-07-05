module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<f32>
    %3 = stablehlo.log %0 : tensor<f32>
    %4 = stablehlo.multiply %2, %3 : tensor<f32>
    %5 = stablehlo.subtract %arg1, %1 : tensor<f32>
    %6 = stablehlo.subtract %1, %0 : tensor<f32>
    %7 = stablehlo.log %6 : tensor<f32>
    %8 = stablehlo.multiply %5, %7 : tensor<f32>
    %9 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %14 = chlo.lgamma %13 : tensor<f32> -> tensor<f32>
    %15 = stablehlo.add %4, %8 : tensor<f32>
    %16 = stablehlo.add %10, %12 : tensor<f32>
    %17 = stablehlo.add %16, %14 : tensor<f32>
    %18 = stablehlo.add %15, %17 : tensor<f32>
    return %18 : tensor<f32>
  }
}
