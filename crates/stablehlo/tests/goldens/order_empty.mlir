module {
  func.func @logdensity(%arg0: tensor<0xf32>, %arg1: tensor<0xi1>) -> (tensor<f32>, tensor<0xf32>, tensor<i1>, tensor<i1>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<false> : tensor<i1>
    %2 = stablehlo.reduce(%arg1 init: %1) applies stablehlo.or across dimensions = [0] : (tensor<0xi1>, tensor<i1>) -> tensor<i1>
    %3 = stablehlo.constant dense<true> : tensor<i1>
    %4 = stablehlo.reduce(%arg1 init: %3) applies stablehlo.and across dimensions = [0] : (tensor<0xi1>, tensor<i1>) -> tensor<i1>
    return %0, %arg0, %2, %4 : tensor<f32>, tensor<0xf32>, tensor<i1>, tensor<i1>
  }
}
