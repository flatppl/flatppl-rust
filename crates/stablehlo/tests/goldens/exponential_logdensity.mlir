module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %4 = stablehlo.log %arg0 : tensor<f32>
    %5 = stablehlo.multiply %arg0, %0 : tensor<f32>
    %6 = stablehlo.negate %5 : tensor<f32>
    %7 = stablehlo.add %4, %6 : tensor<f32>
    return %7 : tensor<f32>
  }
}
