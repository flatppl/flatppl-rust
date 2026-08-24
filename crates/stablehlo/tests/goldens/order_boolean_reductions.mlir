module {
  func.func @logdensity(%arg0: tensor<4xi1>) -> (tensor<i1>, tensor<i1>) {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    %2 = stablehlo.constant dense<true> : tensor<i1>
    %3 = stablehlo.reduce(%arg0 init: %2) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    return %1, %3 : tensor<i1>, tensor<i1>
  }
}
