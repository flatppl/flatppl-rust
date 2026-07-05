module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.log %arg1 : tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %4 = stablehlo.negate %3 : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<f32>
    %7 = stablehlo.log %0 : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.multiply %arg1, %0 : tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.add %2, %4 : tensor<f32>
    %12 = stablehlo.add %11, %8 : tensor<f32>
    %13 = stablehlo.add %12, %10 : tensor<f32>
    return %13 : tensor<f32>
  }
}
