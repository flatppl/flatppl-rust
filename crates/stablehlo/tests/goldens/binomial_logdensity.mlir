module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.add %arg0, %1 : tensor<f32>
    %3 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %4 = stablehlo.add %0, %1 : tensor<f32>
    %5 = chlo.lgamma %4 : tensor<f32> -> tensor<f32>
    %6 = stablehlo.negate %5 : tensor<f32>
    %7 = stablehlo.subtract %arg0, %0 : tensor<f32>
    %8 = stablehlo.add %7, %1 : tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.add %3, %6 : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    %13 = stablehlo.log %arg1 : tensor<f32>
    %14 = stablehlo.multiply %0, %13 : tensor<f32>
    %15 = stablehlo.subtract %1, %arg1 : tensor<f32>
    %16 = stablehlo.log %15 : tensor<f32>
    %17 = stablehlo.multiply %7, %16 : tensor<f32>
    %18 = stablehlo.add %12, %14 : tensor<f32>
    %19 = stablehlo.add %18, %17 : tensor<f32>
    return %19 : tensor<f32>
  }
}
