module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.multiply %1, %arg1 : tensor<f32>
    %3 = stablehlo.log %2 : tensor<f32>
    %4 = stablehlo.negate %3 : tensor<f32>
    %5 = stablehlo.subtract %0, %arg0 : tensor<f32>
    %6 = stablehlo.abs %5 : tensor<f32>
    %7 = stablehlo.divide %6, %arg1 : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.add %4, %8 : tensor<f32>
    return %9 : tensor<f32>
  }
}
