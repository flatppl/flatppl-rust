module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.add %0, %arg1 : tensor<f32>
    %2 = chlo.lgamma %1 : tensor<f32> -> tensor<f32>
    %3 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %4 = stablehlo.negate %3 : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.add %0, %5 : tensor<f32>
    %7 = chlo.lgamma %6 : tensor<f32> -> tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.add %2, %4 : tensor<f32>
    %10 = stablehlo.add %9, %8 : tensor<f32>
    %11 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %12 = stablehlo.log %11 : tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.log %arg0 : tensor<f32>
    %15 = stablehlo.add %14, %13 : tensor<f32>
    %16 = stablehlo.multiply %0, %15 : tensor<f32>
    %17 = stablehlo.log %arg1 : tensor<f32>
    %18 = stablehlo.add %17, %13 : tensor<f32>
    %19 = stablehlo.multiply %arg1, %18 : tensor<f32>
    %20 = stablehlo.add %10, %16 : tensor<f32>
    %21 = stablehlo.add %20, %19 : tensor<f32>
    return %21 : tensor<f32>
  }
}
