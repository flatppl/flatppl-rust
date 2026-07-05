module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.add %0, %arg0 : tensor<f32>
    %2 = chlo.lgamma %1 : tensor<f32> -> tensor<f32>
    %3 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %4 = stablehlo.negate %3 : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.add %0, %5 : tensor<f32>
    %7 = chlo.lgamma %6 : tensor<f32> -> tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.add %2, %4 : tensor<f32>
    %10 = stablehlo.add %9, %8 : tensor<f32>
    %11 = stablehlo.log %arg1 : tensor<f32>
    %12 = stablehlo.add %arg1, %5 : tensor<f32>
    %13 = stablehlo.log %12 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.add %11, %14 : tensor<f32>
    %16 = stablehlo.multiply %arg0, %15 : tensor<f32>
    %17 = stablehlo.multiply %0, %13 : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.add %10, %16 : tensor<f32>
    %20 = stablehlo.add %19, %18 : tensor<f32>
    return %20 : tensor<f32>
  }
}
