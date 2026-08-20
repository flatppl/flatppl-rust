module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> tensor<f32> {
    %0 = stablehlo.abs %arg0 : tensor<4xf32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
}
