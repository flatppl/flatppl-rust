module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<4.0> : tensor<f32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %4 = stablehlo.log %3 : tensor<f32>
    %5 = stablehlo.multiply %0, %4 : tensor<f32>
    %6 = stablehlo.add %1, %5 : tensor<f32>
    return %6 : tensor<f32>
  }
}
