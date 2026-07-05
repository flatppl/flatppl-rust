module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.multiply %arg0, %0 : tensor<f32>
    %3 = stablehlo.negate %2 : tensor<f32>
    %4 = stablehlo.add %1, %3 : tensor<f32>
    return %4 : tensor<f32>
  }
}
