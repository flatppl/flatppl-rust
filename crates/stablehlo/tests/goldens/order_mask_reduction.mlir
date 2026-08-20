module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<i1>, tensor<i1>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.compare GT, %arg0, %1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %3 = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%2 init: %3) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.constant dense<3.0> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.compare GT, %arg0, %6 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %8 = stablehlo.constant dense<true> : tensor<i1>
    %9 = stablehlo.reduce(%7 init: %8) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    return %4, %9 : tensor<i1>, tensor<i1>
  }
}
