module {
  func.func @logdensity(%arg0: tensor<0xf32>, %arg1: tensor<0xi1>) -> (tensor<f32>, tensor<0xf32>, tensor<i1>, tensor<i1>, tensor<0xf32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<false> : tensor<i1>
    %2 = stablehlo.constant dense<true> : tensor<i1>
    return %0, %arg0, %1, %2, %arg0 : tensor<f32>, tensor<0xf32>, tensor<i1>, tensor<i1>, tensor<0xf32>
  }
}
