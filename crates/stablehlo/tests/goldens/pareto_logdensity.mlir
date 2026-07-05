module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.log %arg1 : tensor<f32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.add %arg0, %4 : tensor<f32>
    %6 = stablehlo.log %0 : tensor<f32>
    %7 = stablehlo.multiply %5, %6 : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.add %1, %3 : tensor<f32>
    %10 = stablehlo.add %9, %8 : tensor<f32>
    return %10 : tensor<f32>
  }
}
