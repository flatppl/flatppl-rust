module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.subtract %3, %0 : tensor<f32>
    %5 = stablehlo.subtract %3, %arg0 : tensor<f32>
    %6 = stablehlo.log %5 : tensor<f32>
    %7 = stablehlo.multiply %4, %6 : tensor<f32>
    %8 = stablehlo.add %2, %7 : tensor<f32>
    return %8 : tensor<f32>
  }
}
